# Summarization-Tool-API
Obtain a Hugging Face at https://huggingface.co/settings/tokens and add it to `config.py` for faster summarization inference

If you prefer not to use the Hugging Face API, it is recommended that you run the `download_models.py` script to locally download the prediction models as to prevent them from being downloaded from the transformers library every time the API is reloaded.

Run the following command in the root repository path to run the api
```
uvicorn main:app
```

You can test your token by running the file `huggingface_api_test.py`

The file `summarization.py` contains the logic of the api

The file `main.py` contains the request structure and the route declaration

The file `parallel_test.py` contains a file to test the performance of the summarization script while running sequentially in comparison to a parallel approach.