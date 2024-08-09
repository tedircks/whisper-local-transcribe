# My non-GUI version of the app 
import os
import datetime
import whisper
import configparser
import argparse
from torch import cuda, Generator
import colorama
from colorama import Back,Fore
colorama.init(autoreset=True)

# Initialize the config parser and read the config file
config = configparser.ConfigParser()
config.read('config.ini')

# Constants
ARCHIVE_PATH = config['DEFAULT']['ARCHIVE_PATH']
TRANSCRIPTIONS_PATH = config['DEFAULT']['TRANSCRIPTIONS_PATH']
VIDEO_PATH = config['DEFAULT']['VIDEO_PATH']

# Main function
def transcribe(filename, model=None, language=None, verbose=False):
    """
    Transcribes audio files in a specified folder using OpenAI's Whisper model.

    Args:
        path (str): Path to the folder containing the audio files.
        glob_file (list): List of audio file paths to transcribe.
        model (str, optional): Name of the Whisper model to use for transcription.
            Defaults to None, which uses the default model.
        language (str, optional): Language code for transcription. Defaults to None,
            which enables automatic language detection.
        verbose (bool, optional): If True, enables verbose mode with detailed information
            during the transcription process. Defaults to False.

    Returns:
        str: A message indicating the result of the transcription process.

    Raises:
        RuntimeError: If an invalid file is encountered, it will be skipped.

    Notes:
        - The function downloads the specified model if not available locally.
        - The transcribed text files will be saved in a "transcriptions" folder
          within the specified path.

    """    
    # Setup
    # Check for GPU acceleration
    if cuda.is_available():
        Generator('cuda').manual_seed(42)
    else:
        Generator().manual_seed(42)

    model = whisper.load_model(model)

    input_file_path = os.path.join(VIDEO_PATH, filename)
    output_file_path = os.path.join(TRANSCRIPTIONS_PATH, filename.split('.')[0] + ".txt")

    print(input_file_path)

    # Trnascribe
    print(Back.CYAN + '\nTrying to transcribe file: {}\U0001f550'.format(input_file_path))
    try:
        result = model.transcribe(
            input_file_path, 
            language=language, 
            verbose=verbose
            )
        
        # Make folder if missing 
        print(Back.CYAN + '\nChecking output directory')
        try:
            os.makedirs(TRANSCRIPTIONS_PATH, exist_ok=True)
        except FileExistsError:
            pass

        # Create segments for text files
        start = []
        end = []
        text = []
        print(Back.CYAN + '\nParsing model results')
        for segment in result['segments']:
            start.append(str(datetime.timedelta(seconds=segment['start'])))
            end.append(str(datetime.timedelta(seconds=segment['end'])))
            text.append(segment['text'])

        # Save files to transcriptions folder
        print(Back.CYAN + '\nWriting output')
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(filename)
            for i in range(len(result['segments'])):
                file.write('\n[{} --> {}]:{}'.format(start[i], end[i], text[i]))

        # move mp4 to archive
        os.rename(os.path.join(VIDEO_PATH, filename), os.path.join(ARCHIVE_PATH, filename))
        
    except RuntimeError:
        print(Fore.RED + 'Not a valid file, skipping.')
        pass

    print(Back.CYAN + '\nComplete')

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files.")
    
    parser.add_argument("input_file_path", type=str, help="Path to the audio file or directory containing audio files.")
    parser.add_argument("--model", type=str, default="small", help="Model to use for transcription.")
    parser.add_argument("--language", type=str, default="english", help="Language of the audio.")
    parser.add_argument("--verbose", type=bool, default=False, help="Increase output verbosity.")
    
    args = parser.parse_args()
    
    transcribe(args.input_file_path, args.model, args.language, args.verbose)

if __name__ == "__main__":
    main()