## Overview

Adversarial examples are specialised inputs created with the purpose of confusing a neural network, resulting in the misclassification of a given input. 

These notorious inputs are indistinguishable to the human eye, but cause the network to fail to identify the contents of the image. 

There are several types of such attacks, however, here the focus is on the fast gradient sign method attack, which is a white box attack whose goal is to ensure misclassification. 

A white box attack is where the attacker has complete access to the model being attacked. One of the most famous examples of an adversarial image shown below is taken from the aforementioned paper.

![Adversary attack](https://github.com/SalahMouslih/Adversary-attacks/blob/main/adversarial_example.png)


## Testing code

In the project directory, you can run:

### `uvicorn main:app --reload --host 0.0.0.0`

Runs the FastAPI Server using uvicorn\
Your API will be running at  [http://0.0.0.0:8000](http://localhost:8000)

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.


