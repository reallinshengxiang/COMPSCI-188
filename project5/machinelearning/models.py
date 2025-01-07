from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        self.w = Parameter(ones(1, dimensions))

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

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        score = matmul(self.w, x.T).squeeze()
        return score.unsqueeze(0)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if self.run(x) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            while True:
                flag = True
                for batch in dataloader:
                    x = batch['x'].squeeze(0)
                    y = batch['label'].item()
                    predict_y = self.get_prediction(x)
                    if predict_y != y:
                        self.w += y * x
                        flag = False
                if flag:
                    return



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
        self.layer1 = Linear(1, 128)
        self.layer2 = Linear(128, 128)
        self.layer3 = Linear(128, 1)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        return self.layer3(x)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.forward(x)
        return mse_loss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        batch_size = 32
        epochs = 500
        learning_rate = 0.001
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                x_batch = batch['x'].unsqueeze(1)
                y_batch = batch['label'].unsqueeze(1)
                optimizer.zero_grad()
                loss = self.get_loss(x_batch, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            #if epoch % 100 == 0:
            #    print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader)}')



class DigitClassificationModel(Module):
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
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        self.layer1 = Linear(input_size, 1024)
        self.layer2 = Linear(1024, 512)
        self.layer3 = Linear(512, 256)
        self.layer4 = Linear(256, output_size)



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        return self.layer4(x)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        predicted_y = self.run(x)
        return cross_entropy(predicted_y, y)
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        batch_size = 128
        epochs = 10
        learning_rate = 0.001
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                x_batch = batch['x'].view(-1, 28 * 28)
                y_batch = batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x_batch, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader)} ,accuracy: {dataset.get_validation_accuracy()}')
            if dataset.get_validation_accuracy() >= 0.98 :
                return



class LanguageIDModel(Module):
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
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        self.layer1 = Linear(self.num_chars, 128)
        self.layer2 = Linear(128, 64)
        self.layer3 = Linear(64, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = torch.zeros(128)
        for x in xs:
            h = relu(self.layer1(x) + h)
        h = relu(self.layer2(h))
        result = self.layer3(h)
        return result

    
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
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(xs)
        return cross_entropy(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        batch_size = 64
        epochs = 10
        learning_rate = 0.01
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                x_batch, y_batch = batch['x'], batch['label']
                x_batch = movedim(x_batch, 0, 1)
                optimizer.zero_grad()
                loss = self.get_loss(x_batch, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            #print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader)} ，accuracy：{dataset.get_validation_accuracy()}')
            if dataset.get_validation_accuracy() >= 0.85:
                return


def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    Output_Tensor = torch.zeros((input_tensor_dimensions[0] - weight_dimensions[0] + 1, input_tensor_dimensions[1] - weight_dimensions[1] + 1))
    for y in range(input_tensor_dimensions[0] - weight_dimensions[0] + 1):
        for x in range(input_tensor_dimensions[1] - weight_dimensions[1] + 1):
            Output_Tensor[y, x]= torch.sum(input[y : y + weight_dimensions[0], x : x + weight_dimensions[1]] * weight)
    "*** End Code ***"
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10
        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        input_size = 26 * 26
        self.layer1 = Linear(input_size, 1024)
        self.layer2 = Linear(1024, 512)
        self.layer3 = Linear(512, 256)
        self.layer4 = Linear(256, output_size)


    def run(self, x):
        return self(x)
 
    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        #x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = torch.stack([Convolve(sample, self.convolution_weights) for sample in x])
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        return self.layer4(x)



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        predicted_y = self.forward(x)
        return cross_entropy(predicted_y, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        batch_size = 128
        epochs = 10
        learning_rate = 0.001
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                x_batch, y_batch = batch['x'], batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x_batch, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            #print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader)} ，accuracy：{dataset.get_validation_accuracy()}')
            if dataset.get_validation_accuracy() >= 0.85:
                return


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size,layer_size)

        #Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
       
        self.layer_size = layer_size


    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have 
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:
    
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        """YOUR CODE HERE"""
        Q = self.q_layer(input)
        K = self.k_layer(input)
        V = self.v_layer(input)
        dot_product = torch.matmul(K, Q.transpose(-2, -1)) / (self.layer_size ** 0.5)
        dot_product = dot_product.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))[0]
        attention_weights = softmax(dot_product, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output
