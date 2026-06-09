import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # for chromadb error

import sqlite3
import uuid
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import sqlglot
from sqlglot.optimizer.qualify import qualify
from openrouter import OpenRouter

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRICES_CSV = os.path.join(DATA_DIR, "prices_clean.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews_clean.csv")
FILTER_TABLE = "prices"
LLM_MODEL = "google/gemma-4-31b-it:free" #"openai/gpt-oss-20b:free"
MAX_RESULTS = 5


# ── Initialization ────────────────────────────────────────────────────────────

def initialize():
    """
    Load CSVs, set up SQLite DB, build LLM client, and return shared state dict.
    Call once at app startup and store result in st.session_state.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not found in environment variables.")

    client = OpenRouter(api_key=api_key)
    chroma_client = chromadb.Client()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    df_prices = pd.read_csv(PRICES_CSV)
    df_reviews = pd.read_csv(REVIEWS_CSV)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df_prices.to_sql(FILTER_TABLE, conn, index=True, if_exists="replace")

    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({FILTER_TABLE})")
    columns = cursor.fetchall()
    columns_info = [(col[1], col[2]) for col in columns]
    table_description = "Columns:\n" + "\n".join(
        [f"  - {name}: {col_type}" for name, col_type in columns_info]
    )

    brands = df_prices["brand"].dropna().unique().tolist()
    prod_names = df_prices["name"].dropna().unique().tolist()

    return {
        "llm_client": client,
        "chroma_client": chroma_client,
        "embedding_function": sentence_transformer_ef,
        "conn": conn,
        "cursor": cursor,
        "df_prices": df_prices,
        "df_reviews": df_reviews,
        "columns_info": columns_info,
        "table_description": table_description,
        "brands": brands,
        "prod_names": prod_names,
        "collections": {},   # product_id -> chroma collection
    }


# ── Step 1: Query Decomposition ───────────────────────────────────────────────

def decompose_query(question: str, role: str, answer_pref: str, state: dict) -> dict:
    """
    LLM Call #1: extract structured filters + semantic intent from the question.
    Returns parsed filter JSON dict.
    """
    json_format = """
    {
    "filters": {
        "column_name1": "column_value1",
        "column_name2": "column_value2"
    },
    "semantic_intent": "...",
    "original_query": "..."
    }
    """
    json_output_example = """
    {
    "filters": {
        "brand": "Samsung",
        "amount": {"operator": "<", "value": 500}
    },
    "semantic_intent": "battery issues, battery complaints",
    "original_query": "Which Samsung phones under $500 have battery issues?"
    }
    """
    prompt = f"""
    You are an information extraction system.

    Your task is to analyze a business question for a {role} and extract:
    1. Structured filters (ONLY for the {FILTER_TABLE} table)
    2. Semantic intent (for review search)

    Schema of {FILTER_TABLE}:
    {state['table_description']}

    Brands: {state['brands']}
    Product Names: {state['prod_names']}

    Rules:
    - Only extract filters that map directly to the schema
    - Only use the brands and product names provided here as filter
    - Do NOT infer values not explicitly mentioned
    - Keep semantic intent focused on {role} and {answer_pref} (e.g., Summary based on battery, durability, complaints)
    - If a filter is not present, omit it

    Return STRICT JSON:
    {json_format}

    Example:
    Query: "Which Samsung phones under $500 have battery issues?"
    Output:
    {json_output_example}

    Now process:
    {question}
    """.strip()

    response = state["llm_client"].chat.send(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── Step 2: SQL Generation & Execution ───────────────────────────────────────

def generate_sql(filter_json: dict, state: dict) -> str:
    """
    LLM Call #2: generate a SQL query from the filter JSON.
    Returns the raw SQL string from the LLM.
    """
    sql_json_example = """
    "filters": {
        "brand": "Samsung",
        "amount": {"operator": "<", "value": 500}
    }
    """
    prompt = f"""
    You are a SQL generation system.

    Generate a SQL query to retrieve product IDs from the {FILTER_TABLE} table.

    Schema of {FILTER_TABLE}:
    {state['table_description']}

    Rules:
    - ONLY generate a SELECT query
    - ONLY return the column: id
    - DO NOT include explanations or markdown
    - DO NOT hallucinate columns
    - Use WHERE clauses only if filters exist
    - Use correct SQLite syntax

    Input JSON:
    {json.dumps(filter_json, indent=2)}

    Output:
    SQL query only.

    Example:
    Input:
    {sql_json_example}

    Output:
    SELECT id FROM prices WHERE brand = 'Samsung' AND amount < 500;

    Now generate SQL:
    """.strip()

    response = state["llm_client"].chat.send(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.replace("```", "").strip()

    return raw


def validate_and_execute_sql(llm_sql: str, state: dict) -> tuple[str, list]:
    """
    Validate SQL with sqlglot, execute against SQLite, return (validated_sql, rows).
    Raises ValueError if validation fails.
    """
    q = ""
    for name, col_type in state["columns_info"]:
        q += f'"{name}": "{col_type}",'
    q = q[:-1]
    schema_str = "{" + f'"{FILTER_TABLE}"' + ": {" + q + "}}"
    schema = json.loads(schema_str)

    expression = sqlglot.parse_one(llm_sql, read="sqlite")
    qualified = qualify(expression, schema=schema, dialect="sqlite")
    validated_sql = qualified.sql(dialect="sqlite")

    state["cursor"].execute(validated_sql)
    rows = state["cursor"].fetchall()
    return validated_sql, rows


# ── Step 3: Get Filtered Reviews DataFrame ───────────────────────────────────

def get_filtered_reviews(rows: list, state: dict) -> pd.DataFrame:
    """
    Given SQL result rows (each row is (id,)), return all matching reviews as a DataFrame.
    """
    product_ids = [r[0] for r in rows]
    df = state["df_reviews"][state["df_reviews"]["id"].isin(product_ids)].copy()
    return df


# ── Step 4: Vectorize Reviews ─────────────────────────────────────────────────

def vectorize_reviews(product_id: str, df_reviews_filtered: pd.DataFrame, state: dict):
    """
    Create (or reuse) a ChromaDB collection keyed by product_id.
    Returns the collection.
    """

    print(state["collections"])
    collection_name = f"reviews_{str(product_id).replace('-', '_').replace(' ', '_')[:40]}"

    if product_id in state["collections"]:
        return state["collections"][product_id]

    try:
        collection = state["chroma_client"].get_collection(
            name=collection_name,
            embedding_function=state['embedding_function']
        )
        state["collections"][product_id] = collection
        return collection
    except Exception:
        collection = state["chroma_client"].create_collection(
            name=collection_name,
            embedding_function=state['embedding_function']
        )
        # only add documents when freshly created

    df_target = df_reviews_filtered[df_reviews_filtered["id"] == product_id].copy()
    df_target = df_target.dropna(subset=["reviews.text"])

    if df_target.empty:
        state["collections"][product_id] = collection
        return collection

    metadata_cols = [
        "reviews.date",
        "reviews.doRecommend",
        "reviews.numHelpful",
        "reviews.rating",
        "reviews.title",
        "reviews.username",
    ]
    available_meta_cols = [c for c in metadata_cols if c in df_target.columns]

    meta_df = df_target[available_meta_cols].copy()
    # ChromaDB requires all metadata values to be str/int/float/bool
    for col in meta_df.columns:
        meta_df[col] = meta_df[col].fillna("").astype(str)

    collection.add(
        ids=[str(uuid.uuid4()) for _ in range(len(df_target))],
        documents=df_target["reviews.text"].tolist(),
        metadatas=meta_df.to_dict(orient="records"),
    )

    state["collections"][product_id] = collection
    return collection


# ── Step 5: Rewrite Semantic Query ────────────────────────────────────────────

def rewrite_semantic_query(question: str, filter_json: dict, role: str, answer_pref: str, state: dict) -> str:
    """
    LLM Call #3: rewrite the question into a clean semantic search phrase.
    """
    prompt = f"""
    You are a query rewriting system for semantic search.

    Your task:
    - Remove all structured filtering conditions already handled by SQL
    - Keep ONLY the part relevant for semantic similarity search in reviews
    - Expand slightly for better retrieval (add synonyms if useful)
    - Keep it concise (1 short sentence or phrase)

    Rules:
    - Do NOT include brand, price, color, year, etc. if already used as filters
    - Focus on user intent ({role}) and answer preference ({answer_pref})
    - If no semantic intent exists, return: "general product feedback"

    Input:
    Original Query: {question}
    Filters Applied:
    {json.dumps(filter_json, indent=2)}

    Output:
    Rewritten semantic query only. No explanation, no preamble.
    """.strip()

    response = state["llm_client"].chat.send(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip().strip('"')


# ── Step 6: Vector Retrieval ──────────────────────────────────────────────────

def retrieve_reviews(collection, semantic_query: str) -> list[str]:
    """
    Query the ChromaDB collection. n_results is dynamically capped by collection size.
    Returns list of review text strings.
    """
    count = collection.count()
    if count == 0:
        return []

    n = min(MAX_RESULTS, count)
    results = collection.query(query_texts=[semantic_query], n_results=n)
    return results["documents"][0]


# ── Step 7: Final Answer Generation ──────────────────────────────────────────

def generate_answer(
        question: str, 
        semantic_query: str, 
        retrieved_reviews: list[str], 
        role: str, 
        answer_pref: str, 
        state: dict
    ) -> str:
    """
    LLM Call #4: synthesize a business-analyst-style answer from retrieved reviews.
    """
    review_block = "\n--> ".join(retrieved_reviews) if retrieved_reviews else "No reviews available."

    prompt = f"""
    You are a business analyst.

    Answer the user's ({role}) question using ONLY the provided review data.
    Keep in mind the answer preference: {answer_pref}.

    Rules:
    - Base your answer strictly on the reviews
    - Ensure to answer the points mentioned in the semantic query
    - Do NOT hallucinate
    - If evidence is weak or insufficient, say so
    - Summarize patterns across reviews
    - Highlight common positives/negatives

    Input:

    User Question:
    {question}

    Semantic query:
    {semantic_query}

    Filtered Context (reviews):
    {review_block}

    Answer:
    """.strip()

    response = state["llm_client"].chat.send(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_query(question: str, role: str, answer_pref: str, state: dict) -> dict:
    """
    Full pipeline for one question. Returns a result dict with all intermediate
    outputs for display. Raises NoRowsError if SQL returns no results.
    """
    filter_json = decompose_query(question, role, answer_pref, state)
    llm_sql = generate_sql(filter_json, state)
    validated_sql, rows = validate_and_execute_sql(llm_sql, state)

    if not rows:
        raise NoRowsError(
            filter_json=filter_json,
            validated_sql=validated_sql,
            brands=state["brands"],
            prod_names=state["prod_names"],
        )

    df_filtered = get_filtered_reviews(rows, state)

    # Use the first product id for vectorization
    primary_product_id = rows[0][0]
    collection = vectorize_reviews(primary_product_id, df_filtered, state)

    semantic_query = rewrite_semantic_query(question, filter_json, role, answer_pref, state)
    retrieved_reviews = retrieve_reviews(collection, semantic_query)
    answer = generate_answer(question, semantic_query, retrieved_reviews, role, answer_pref, state)

    return {
        "question": question,
        "filter_json": filter_json,
        "validated_sql": validated_sql,
        "rows": rows,
        "df_filtered": df_filtered,
        "semantic_query": semantic_query,
        "retrieved_reviews": retrieved_reviews,
        "answer": answer,
        "role": role,
        "answer_pref": answer_pref,
    }


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class NoRowsError(Exception):
    def __init__(self, filter_json, validated_sql, brands, prod_names):
        self.filter_json = filter_json
        self.validated_sql = validated_sql
        self.brands = brands
        self.prod_names = prod_names
        super().__init__("SQL query returned no results.")
