import json
import csv
import textwrap
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def split_description(text, width=100):
    logger.debug(f"Splitting description of length {len(text)} characters")
    parts = textwrap.wrap(text, width=width)
    logger.debug(f"Split into {len(parts)} parts")
    return parts

def json_to_csv_with_split_description(input_json_file, output_csv_file):
    start_time = datetime.now()
    logger.info(f"Starting conversion from {input_json_file} to {output_csv_file}")
    
    try:
        # Read JSON file
        logger.info("Reading JSON file...")
        with open(input_json_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON with {len(data)} scenarios")
        
        # Open CSV file for writing
        logger.info("Creating CSV file...")
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = [
                'game_name', 'scenario', 'description_part', 
                'participant1_name', 'participant1_profile',
                'participant2_name', 'participant2_profile',
                'choice_a', 'choice_b',
                'is_last_row'
            ]
            writer.writerow(headers)
            logger.info(f"Wrote CSV headers: {', '.join(headers)}")
            
            # Process each scenario
            total_rows = 0
            for i, item in enumerate(data, 1):
                logger.info(f"Processing scenario {i}/{len(data)}: {item['scenario']}")
                
                # Split description into parts
                desc_parts = split_description(item['description'])
                logger.debug(f"Description split into {len(desc_parts)} parts")
                
                # Write each part as a separate row
                for j, desc_part in enumerate(desc_parts):
                    is_last_row = '1' if j == len(desc_parts) - 1 else '0'
                    choice_a = str(list(item['behavior_choices'].items())[0])
                    choice_b = str(list(item['behavior_choices'].items())[1])
                    row = [
                        item['game_name'],
                        item['scenario'],
                        desc_part,
                        item['participants'][0]['name'],
                        item['participants'][0]['profile'],
                        item['participants'][1]['name'],
                        item['participants'][1]['profile'],
                        choice_a, 
                        choice_b,
                        is_last_row
                    ]
                    writer.writerow(row)
                    total_rows += 1
                
                logger.debug(f"Wrote {len(desc_parts)} rows for scenario '{item['scenario']}'")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Conversion completed successfully!")
        logger.info(f"Raw Json file: {input_json_file}")
        logger.info(f"Output CSV file: {output_csv_file}")
        logger.info(f"Total scenarios processed: {len(data)}")
        logger.info(f"Total rows written: {total_rows}")
        logger.info(f"Average rows per scenario: {total_rows/len(data):.2f}")
        logger.info(f"Processing time: {duration:.2f} seconds")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e} at {item['scenario']}")
        raise

# Use the function
if __name__ == "__main__":
    input_file = 'group_chat/all_scenarios.json'
    output_file = 'group_chat/all_scenarios4annotation.csv'
    
    try:
        json_to_csv_with_split_description(input_file, output_file)
    except Exception as e:
        logger.error(f"Script execution failed: {e}")