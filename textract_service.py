# textract_service.py
"""
Service module for AWS Textract integration and document post-processing
"""

import logging
import webbrowser, os
import json
import boto3
import io
from io import BytesIO
import sys
from pprint import pprint
import csv
from pdf2image import convert_from_path
from config import AWS_PROFILE, AWS_REGION

class TextractService:
    def __init__(self):
        pass


    @staticmethod
    def _get_rows_columns_map(table_result, blocks_map):
        rows = {}
        scores = []
        for relationship in table_result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    cell = blocks_map[child_id]
                    if cell['BlockType'] == 'CELL':
                        row_index = cell['RowIndex']
                        col_index = cell['ColumnIndex']
                        if row_index not in rows:
                            # create new row
                            rows[row_index] = {}
                        
                        # get confidence score
                        scores.append(str(cell['Confidence']))
                            
                        # get the text value
                        rows[row_index][col_index] = TextractService._get_text(cell, blocks_map)
        return rows, scores


    @staticmethod
    def _get_text(result, blocks_map):
        text = ''
        if 'Relationships' in result:
            for relationship in result['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        word = blocks_map[child_id]
                        if word['BlockType'] == 'WORD':
                            text += word['Text'] + ' '
                        if word['BlockType'] == 'SELECTION_ELEMENT':
                            if word['SelectionStatus'] =='SELECTED':
                                text += 'X '
        return text.strip()


    @staticmethod
    def _get_table_csv_results(file_name):
        with open(file_name, 'rb') as file:
            img_test = file.read()
            bytes_test = bytearray(img_test)
            print('Image loaded', file_name)

        # process using image bytes
        # get the results
        session = boto3.Session(profile_name=AWS_PROFILE)
        client = session.client('textract', region_name=AWS_REGION)
        response = client.analyze_document(Document={'Bytes': bytes_test}, FeatureTypes=['TABLES', 'FORMS'])

        # Get the text blocks
        blocks = response['Blocks']

        blocks_map = {}
        table_blocks = []
        for block in blocks:
            blocks_map[block['Id']] = block
            if block['BlockType'] == "TABLE":
                table_blocks.append(block)

        if len(table_blocks) <= 0:
            return None

        csv_output = ''
        for index, table in enumerate(table_blocks):
            csv_output += TextractService._generate_table_csv(table, blocks_map, index + 1)
            csv_output += '\n\n'

        return csv_output


    @staticmethod
    def _generate_table_csv(table_result, blocks_map, table_index):
        rows, scores = TextractService._get_rows_columns_map(table_result, blocks_map)

        table_id = 'Table_' + str(table_index)
        
        # Use StringIO to properly write CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        
        # Write table header
        output.write('Table: {0}\n\n'.format(table_id))
        
        # Write table data
        for row_index in sorted(rows.keys()):
            cols = rows[row_index]
            row_data = []
            for col_index in sorted(cols.keys()):
                row_data.append(cols[col_index])
            writer.writerow(row_data)
        
        # Write confidence scores
        # output.write('\n\nConfidence Scores % (Table Cell)\n')
        
        # # Determine number of columns
        # if rows:
        #     first_row = rows[min(rows.keys())]
        #     col_indices = len(first_row)
        # else:
        #     col_indices = 0
        
        # # Write scores in rows matching table structure
        # score_idx = 0
        # while score_idx < len(scores):
        #     score_row = []
        #     for _ in range(col_indices):
        #         if score_idx < len(scores):
        #             score_row.append(scores[score_idx])
        #             score_idx += 1
        #         else:
        #             break
        #     writer.writerow(score_row)
        
        output.write('\n\n')
        
        return output.getvalue()


    @staticmethod
    def _process_pdf(pdf_file):
        """
        Convert PDF pages to images and process each page
        """
        print(f"Converting PDF to images: {pdf_file}")
        
        # Convert PDF to images (one image per page)
        images = convert_from_path(pdf_file, dpi=300)  # Higher DPI for better quality
        
        print(f"PDF has {len(images)} page(s)")
        
        all_csv_output = ''
        temp_files = []
        
        for i, image in enumerate(images):
            # Save each page as a temporary image
            temp_image_path = f'temp_page_{i+1}.png'
            image.save(temp_image_path, 'PNG')
            temp_files.append(temp_image_path)
            
            print(f"\nProcessing page {i+1}...")
            
            # Process the image with Textract
            csv_output = TextractService._get_table_csv_results(temp_image_path)
            
            if csv_output:
                # all_csv_output += f"=== PAGE {i+1} ===\n\n"
                all_csv_output += csv_output
                # all_csv_output += '\n\n'
            else:
                print(f"No tables found on page {i+1}")
            
            # break
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"Cleaned up: {temp_file}")
            except:
                pass
        
        return all_csv_output if all_csv_output else "<b> NO Tables FOUND in PDF </b>"


    def process_file(self, file_name):
        # file_extension = os.path.splitext(file_name)[1].lower()
        
        # if file_extension == '.pdf':
        #     # Process PDF by converting to images
        #     table_csv = self._process_pdf(file_name)
        # else:
        #     # Process image directly
        #     table_csv = self._get_table_csv_results(file_name)
        #     if not table_csv:
        #         table_csv = "<b> NO Table FOUND </b>"
        table_csv = ''
        if isinstance(file_name, (list, tuple)):
            for path in file_name:
                try:
                    out_csv = self._get_table_csv_results(path) + '\n\n' 
                except Exception as e:
                    logging.error("Error processing file %s: %s", path, str(e))
                    out_csv = None
                if out_csv:
                    table_csv += out_csv
                else:
                    logging.info("No Tables detected in %s", path)               
        elif file_name.lower().endswith('.pdf'):
            # Process PDF by converting to images
            try:
                table_csv = self._process_pdf(file_name)
            except Exception as e:
                logging.error("Error processing PDF %s: %s", file_name, str(e))
                table_csv = None
        else:
            # Process image directly
            logging.info("No Tables detected")
            table_csv = None


        output_file = 'test-output.csv'

        # Write to output file
        with open(output_file, "wt") as fout:
            fout.write(table_csv)

        # Show the results
        print('\n' + '='*50)
        print('CSV OUTPUT FILE: ', output_file)
        print('='*50)

        return table_csv
