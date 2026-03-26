from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline   
from src.textSummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline



STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Data Transformation Stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_transformation_pipeline=DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Model Trainer Stage"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} initiated <<<<<<")
    model_trainer=ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
