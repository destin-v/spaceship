
# Decloaking Spaceship

## Note
This was an interesting computer vision problem I completed back in 2021.  It requires building an AI solution to pick out an object from a noisy environment.  I created a novel implementation that was able to train on limited GPU resources.  This code is designed to showcase how to train small models for a difficult problem.  The small models can be stitched together in a *hydra* configuration to provide accurate inferences on many different tasks.

---

**Requirements:**
Python==3.8 is needed in order to install the packages stated in the requirements.txt.

**Problem:**

<img title="spaceship problem" alt="Alt text" src="example.png">

The goal is to detect spaceships which have been fitted with a cloaking device that makes them less visible. You are expected to use a deep learning model to complete this task. The model will take a single channel image as input and detects the spaceship (if it exists). Not all image will contain a spaceship, but they will contain no more than 1. For any spaceship, the model should predict their bounding box and heading. This can be described using five parameters:

* X and Y position (centre of the bounding box)
* Yaw (direction of heading)
* Width (size tangential to the direction of yaw)
* Height (size along the direct of yaw)

We have supplied a base model as a reference which performs poorly and has some serious limitations. You can extend the existing model or reimplement from scratch in any framework of your choice.

The metric for the model is AP at an IOU threshold of 0.7, for at least 1000 random samples, with the default generation parameters (see `main.py`). Please do not modify any of the generation code directly.

**Evaluation Criteria:**
* Model metric, score as high as you can while being under 2 million trainable parameters. Please streamline the parameters where possible
* Model architecture
* Loss function
* Code readability and maintainability, please follow general python conventions

**Deliverables**
1. Report a final score
2. A summary of the model architecture. E.g. `model.summary()` or `torchsummary`
3. A `train.py` script that allows the same model to be reproduced
4. The final model weights
5. A `requirements.txt` file that includes all python dependencies and their versions
6. A `main.py` file that reproduces the reported score


**Tips:**
* Carefully consider how the loss function should be formulated (especially yaw)
* Sanity check how trainable parameters are distributed in the network
* You may use as many training examples as you want. Both train and test used the same generation function
* You may use existing a codebase but please reference the source
* Submitted solutions achieve 0.5 score on average, but it is possible to achieve near perfect score.
* Any pre/post-processing that can be reproduced at inference is fair game.
