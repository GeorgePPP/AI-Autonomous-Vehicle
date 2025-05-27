# import asyncio
# import logging
# from utils import get_api_key
# from chatbot import NDII
# import config

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("NDII-Evaluate")

# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                              
# async def main():
#     # Get API key
#     api_key = get_api_key()
    
#     # Initialize NDII with database
#     logger.info("Creating NDII instance...")
#     ndii = await NDII.create_db(api_key, max_history=4, rag_config=config.RAG)

#     await ndii.evaluate()

# if __name__ == "__main__":
#     asyncio.run(main())