import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        prediction = nn.as_scalar(self.run(x))
        if prediction >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            no_errors = True
            for x, y in dataset.iterate_once(1):
                pred = self.get_prediction(x)
                real = nn.as_scalar(y)
                if pred != real:
                    self.get_weights().update(x, real)
                    no_errors = False
            if no_errors:
                break
        

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.width = 40
        self.desiredLoss = 0.01
        self.batch_size = 10
        self.alpha = 0.03
        
        self.m1 = nn.Parameter(1, self.width)
        self.b1 = nn.Parameter(1, self.width)
        self.m2 = nn.Parameter(self.width, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        layer_1 = nn.Linear(x, self.m1)
        layer_1 = nn.AddBias(layer_1, self.b1)
        layer_1 = nn.ReLU(layer_1)

        output = nn.Linear(layer_1, self.m2)
        output = nn.AddBias(output, self.b2)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                
                grad_m2, grad_b2, grad_m1, grad_b1 = nn.gradients(loss, [self.m2, self.b2, self.m1, self.b1])
                self.m2.update(grad_m2, -self.alpha)
                self.b2.update(grad_b2, -self.alpha)
                self.m1.update(grad_m1, -self.alpha)
                self.b1.update(grad_b1, -self.alpha)

            if nn.as_scalar(loss) < self.desiredLoss:
                break





class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.sizeIN = 784
        self.sizeOUT = 10
        self.width = 80     # width of hidden layer
        self.alpha = 0.3    # learning rate
        self.batch_size = 100
        self.desiredAcc = 0.975
        
        self.m1 = nn.Parameter(self.sizeIN, self.width)
        self.b1 = nn.Parameter(1, self.width)
        self.m2 = nn.Parameter(self.width, self.sizeOUT)
        self.b2 = nn.Parameter(1, self.sizeOUT)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        layer = nn.Linear(x, self.m1)
        layer = nn.AddBias(layer, self.b1)
        layer = nn.ReLU(layer)

        output = nn.Linear(layer, self.m2)
        output = nn.AddBias(output, self.b2)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                acc = dataset.get_validation_accuracy()
                
                grad_m2, grad_b2, grad_m1, grad_b1 = nn.gradients(loss, [self.m2, self.b2, self.m1, self.b1])

                self.m2.update(grad_m2, -self.alpha)
                self.b2.update(grad_b2, -self.alpha)
                self.m1.update(grad_m1, -self.alpha)
                self.b1.update(grad_b1, -self.alpha)

                if acc > self.desiredAcc:
                    return



class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.sizeOUT = len(self.languages)
        self.sizeH = 150     # width of hidden layer
        self.alpha = 0.05    # learning rate
        self.batch_size = 50
        self.desiredAcc = 0.88
        
        self.W_in = nn.Parameter(self.num_chars, self.sizeH)
        self.W_h = nn.Parameter(self.sizeH, self.sizeH)
        self.W_out = nn.Parameter(self.sizeH, self.sizeOUT)

        self.b_in = nn.Parameter(1, self.sizeH)
        self.b_h = nn.Parameter(1, self.sizeH)    
        self.b_out = nn.Parameter(1, self.sizeOUT)    

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        inp = nn.Linear(xs[0], self.W_in)
        inp = nn.AddBias(inp, self.b_in)
        last_h = nn.ReLU(inp)

        for i in range(1, len(xs)):
            next_inp = nn.Linear(xs[i], self.W_in)
            next_inp = nn.AddBias(next_inp, self.b_in)
            h_as_inp = nn.Linear(last_h, self.W_h)
            last_h = nn.Add(next_inp, h_as_inp)
            last_h = nn.AddBias(last_h, self.b_h)
            last_h = nn.ReLU(last_h)

        output = nn.Linear(last_h, self.W_out)
        output = nn.AddBias(output, self.b_out)

        return output


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        predicted_y = self.run(xs)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                acc = dataset.get_validation_accuracy()
                
                grad_W_in, grad_b_in, grad_W_h, grad_b_h, grad_W_out, grad_b_out = \
                            nn.gradients(loss, [self.W_in, self.b_in, self.W_h, self.b_h, self.W_out, self.b_out])

                self.W_in.update(grad_W_in, -self.alpha)
                self.b_in.update(grad_b_in, -self.alpha)
                self.W_h.update(grad_W_h, -self.alpha)
                self.b_h.update(grad_b_h, -self.alpha)
                self.W_out.update(grad_W_out, -self.alpha)
                self.b_out.update(grad_b_out, -self.alpha)

                if acc > self.desiredAcc:
                    return
