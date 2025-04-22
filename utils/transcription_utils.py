import os
import gc
import whisperx

def check_gpu_availability():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def transcribe_audio_with_whisperx(audio_file, model_name="small", device=None, compute_type="float16", batch_size=16):
    if device is None:
        device = check_gpu_availability()
    try:
        print(f"Loading {model_name} model...")
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_file)
        print("Transcribing with Whisper...")
        result = model.transcribe(audio, batch_size=batch_size)
        print(f"Detected language: {result['language']}")
        del model; gc.collect()
        if device == "cuda":
            import torch; torch.cuda.empty_cache()
        print("Aligning words...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        del model_a; gc.collect()
        if device == "cuda":
            import torch; torch.cuda.empty_cache()
        try:
            if os.getenv('HUGGINGFACE_TOKEN'):
                print("Identifying speakers...")
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv('HUGGINGFACE_TOKEN'), device=device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            print(f"Speaker diarization failed: {e}. Continuing without speaker labels.")
        words_with_timestamps = []
        for segment in result["segments"]:
            speaker = segment.get("speaker", "")
            for word in segment.get("words", []):
                if "start" in word and "end" in word:
                    words_with_timestamps.append({
                        "word": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"],
                        "speaker": speaker
                    })
        return words_with_timestamps
    except Exception as e:
        print(f"Error in transcription: {e}")
        raise