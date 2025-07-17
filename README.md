# Deep Learning-Based Lip Reading System

A PyTorch implementation of a lip reading system that can recognize words from video sequences of lip movements. This project includes custom dataset recording, augmentation, and training capabilities.

## Features

- **Custom Dataset Recording**: Record your own lip reading dataset with guided prompts
- **Data Augmentation**: Augment videos with various transformations to increase dataset size
- **Advanced Preprocessing**: Extract lip regions using MediaPipe face detection and landmarks
- **Deep Learning Model**: CNN-RNN architecture for sequence-to-sequence lip reading
- **Real-time Inference**: Test the model with webcam input

## Project Structure

```
self_dataset/
├── scripts/
│   ├── lip_reading_recorder.py    # Record custom dataset
│   ├── augment_dataset.py         # Augment recorded videos
│   ├── process_augmented.py       # Process augmented dataset
│   ├── train_model.py             # Train the lip reading model
│   ├── custom_preprocessor.py     # Process original dataset
│   └── metadata_cleaner.py        # Clean dataset metadata
├── data/                          # Dataset folder (not included)
├── models/                        # Model weights (not included)
├── processed_augmented_dataset/   # Processed data (not included)
└── results/                       # Training results (not included)
```

## Model Architecture

The model consists of:
- **Frontend**: 3D CNN for spatiotemporal feature extraction
- **ResNet**: 2D CNN backbone for spatial features
- **Backend**: Bidirectional GRU for temporal modeling
- **Output**: Linear layer with CTC loss

## Performance

With augmented dataset:
- Training samples: ~1,920
- Validation accuracy: ~60-70%
- Real-time inference: ~20 FPS

## Troubleshooting

1. **MediaPipe warnings**: Normal during initialization, can be ignored
2. **Low extraction rate**: Ensure good lighting and face visibility
3. **CUDA out of memory**: Reduce batch size in training
4. **Slow processing**: Processing 2,400 videos takes time, be patient

## Future Improvements

- [ ] Add more augmentation techniques
- [ ] Implement attention mechanisms
- [ ] Support for phrase-level recognition
- [ ] Multi-language support
- [ ] Mobile deployment

## Acknowledgments

- MediaPipe for face detection and landmarks
- PyTorch for deep learning framework
- Inspired by LipNet and other lip reading research