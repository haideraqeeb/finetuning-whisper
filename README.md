# Fine-Tuning Whisper: Step-by-Step Guide

This guide outlines the steps to reproduce the fine-tuning process for the Whisper model. Ensure you have the necessary dependencies installed and datasets prepared before starting.

---

## Prerequisites

1. **Hardware Requirements**:
   - A GPU-enabled system (recommended: NVIDIA GPU with CUDA support).
   - Sufficient disk space for datasets and model checkpoints.

2. **Software Requirements**:
   - Python 3.8 or later.
   - CUDA toolkit for GPU acceleration (if applicable).
   - Jupyter Notebook or an equivalent environment for running the `.ipynb` file.

3. **Dependencies**:
   Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

4. **Notebook Login**
   Make sure the notebook is logged in using the following code:
   ```python
   from huggingface_hub import notebook_login
   notebook_login
   ```
   After this, just paste a token with *write* access to your hugging face account.
---

## Dataset Preparation

1. **Obtain the Dataset**:
   - Collect a dataset of audio files and corresponding transcriptions.
   - Ensure the dataset is properly split into training, validation, and test sets.

2. **Preprocess the Data**:
   - Make sure audio files to the required format (`.wav`).
   - Tokenize transcriptions using the tokenizer.

3. **Load the Dataset**:
   - Use the Hugging Face `datasets` library to load and preprocess the dataset.
     ```python
     from datasets import load_dataset

     dataset = load_dataset("haideraqeeb/gujarati-asr-16kHz")
     ```

---

## Model Setup

1. **Load the Pre-trained Whisper Model**:
   ```python
   from transformers import WhisperForConditionalGeneration, WhisperProcessor

   model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
   processor = WhisperProcessor.from_pretrained("openai/whisper-small")
   ```

2. **Configure the Model**:
   - Adjust hyperparameters such as learning rate, batch size, and number of training epochs.
   - Freeze specific model layers (if necessary) to optimize training.

---

## Fine-Tuning Process

1. **Prepare the Training Script**:
   - Define the training loop using the `Trainer` API from Hugging Face.
   - Example:
     ```python
     from transformers import TrainingArguments, Trainer

     training_args = Seq2SeqTrainingArguments(
          output_dir="./whisper-gujarati-finetuned",
          per_device_train_batch_size=16,
          gradient_accumulation_steps=1,
          learning_rate=1e-5,
          warmup_steps=500,
          max_steps=4000,
          gradient_checkpointing=True,
          fp16=True,
          evaluation_strategy="steps",
          per_device_eval_batch_size=8,
          predict_with_generate=True,
          generation_max_length=225,
          save_steps=1000,
          eval_steps=1000,
          logging_steps=25,
          report_to=["tensorboard"],
          load_best_model_at_end=True,
          metric_for_best_model="wer",
          greater_is_better=False,
          push_to_hub=True,
     )

     trainer = Seq2SeqTrainer(
          args=training_args,
          model=model,
          train_dataset=dataset["train"],
          eval_dataset=dataset["validation"],
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          tokenizer=processor.feature_extractor,
     )

     trainer.train()
     ```

2. **Monitor Training**:
   - Use logging and callbacks to track training progress.
   - Save checkpoints at regular intervals.

---

## Evaluation

1. **Evaluate on Test Set**:
   - Use the fine-tuned model to generate predictions for the test set.
   - Compute evaluation metrics such as WER (Word Error Rate).
     ```python
     from jiwer import wer

     predictions = trainer.predict(dataset["test"])
     wer_score = wer(actual_transcriptions, predicted_transcriptions)
     print(f"WER: {wer_score}")
     ```

2. **Analyze Results**:
   - Visualize the model’s performance using tools like matplotlib or seaborn.

---

## Exporting the model
   Save the model using the following code. Make sure the notebook login is already done using ```huggingface_hub```. 
   ```python
   model.save_pretrained("your model path")
   processor.save_pretrained("your model path")
   ```