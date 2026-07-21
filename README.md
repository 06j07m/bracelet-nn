# Bracelet Neural Network

Learning how to use Pytorch by making an image classifier for the knots in bracelet patterns.

> [!NOTE]
> A "fun" experiment. Project has been archived.

## Notes

- Custom dataset
    - Scraped 10 images from braceletbook.com
    - Convert to grayscale and cropped out each knot for a total of 1222 images
    - Use cv2 to display images one by one and press keys to label
    - 0 = ↙, 1 = ↘, 2 = ↲, 3 = ↳
    - Split 80% train, 10% validation, 10% test
- Custom model
    - 3 x (convolution > residual block > pooling), then global average and flatten, then 2 x (linear > relu), then softmax
    - residual block: 2 x (convolution > batch norm > relu)
    - Tried pre norm and post norm and did not seem to have much effect
- Pretraining
    - Torchvision MNIST set with 0.5 chance of being inverted
    - 5 epochs
- Training
    - Cross entropy loss
    - 10 epochs
    - Stochastic gradient descent optimizer with learning rate 0.01 and momentum 0.9
    - Exponential learning rate scheduler with gamma 0.9

## Why this setup?

Your guess is as good as mine.

## Does it work?

Not really.