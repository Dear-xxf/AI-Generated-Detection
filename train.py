from GoogleNetV3 import GoogleNetV3Classifier


def main():
    dataset_dir = './dataset'
    classifier = GoogleNetV3Classifier(dataset_dir, batch_size=32, learning_rate=0.001, num_epochs=50)
    classifier.train()
    classifier.save_model('ai_generated_detection.pth')


if __name__ == '__main__':
    main()
