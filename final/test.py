from dotenv import load_dotenv

load_dotenv()

from ragdss import initialize, run_query, NoRowsError

state = initialize()
run_query('Compare the reviews of Kindle Fire HDX 8.9" that are priced below $400 and on sale', state)