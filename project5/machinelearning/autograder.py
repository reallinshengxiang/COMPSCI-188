# A custom autograder for this project

################################################################################
# A mini-framework for autograding
################################################################################

import optparse
import sys
import traceback


class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass

class Tracker(object):
    def __init__(self, questions, maxes, prereqs, mute_output):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.
""".format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        import time
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
          print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
            sum([self.maxes[q] for q in self.questions])))

        print("""
Your grades are NOT yet registered.  To register your grades, make sure
to follow your instructor's guidelines to receive credit on your project.
""")

    def add_points(self, pts):
        self.points[self.current_question] += pts

TESTS = []
PREREQS = {}
def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)

def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn
    return deco

def parse_options(argv):
    parser = optparse.OptionParser(description = 'Run public tests on student code')
    parser.set_defaults(
        edx_output=False,
        gs_output=False,
        no_graphics=False,
        mute_output=False,
        check_dependencies=False,
        )
    parser.add_option('--edx-output',
                        dest = 'edx_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--gradescope-output',
                        dest = 'gs_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--question', '-q',
                        dest = 'grade_question',
                        default = None,
                        help = 'Grade only one question (e.g. `-q q1`)')
    parser.add_option('--no-graphics',
                        dest = 'no_graphics',
                        action = 'store_true',
                        help = 'Do not display graphics (visualizing your implementation is highly recommended for debugging).')
    parser.add_option('--mute',
                        dest = 'mute_output',
                        action = 'store_true',
                        help = 'Mute output from executing tests')
    parser.add_option('--check-dependencies',
                        dest = 'check_dependencies',
                        action = 'store_true',
                        help = 'check that numpy and matplotlib are installed')
    (options, args) = parser.parse_args(argv)
    return options

def main():
    options = parse_options(sys.argv)
    if options.check_dependencies:
        check_dependencies()
        return

    if options.no_graphics:
        disable_graphics()

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, options.mute_output)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()

################################################################################
# Tests begin here
################################################################################

import numpy as np
import matplotlib
import contextlib

from torch import nn, Tensor
import torch
import backend

def check_dependencies():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    line, = ax.plot([], [], color="black")
    plt.show(block=False)

    for t in range(400):
        angle = t * 0.05
        x = np.sin(angle)
        y = np.cos(angle)
        line.set_data([x,-x], [y,-y])
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1e-3)

def disable_graphics():
    backend.use_graphics = False

@contextlib.contextmanager
def no_graphics():
    old_use_graphics = backend.use_graphics
    backend.use_graphics = False
    yield
    backend.use_graphics = old_use_graphics

def verify_node(node, expected_type, expected_shape, method_name):
    if expected_type == 'parameter':
        assert node is not None, (
            "{} should return an instance of nn.Parameter, not None".format(method_name))
        assert isinstance(node, nn.Parameter), (
            "{} should return an instance of nn.Parameter, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'loss':
        assert node is not None, (
            "{} should return an instance a loss node, not None".format(method_name))
        assert isinstance(node, (nn.modules.loss._Loss)), (
            "{} should return a loss node, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'tensor':
        assert node is not None, (
            "{} should return a node object, not None".format(method_name))
        assert isinstance(node, Tensor), (
            "{} should return a node object, instead got type {!r}".format(
            method_name, type(node).__name__))
    else:
        assert False, "If you see this message, please report a bug in the autograder"

    if expected_type != 'loss':
        assert all([(expected is '?' or actual == expected) for (actual, expected) in zip(node.detach().numpy().shape, expected_shape)]), (
            "{} should return an object with shape {}, got {}".format(
                method_name, expected_shape, node.shape))

@test('q1', points=6)
def check_perceptron(tracker):
    import models

    print("Sanity checking perceptron...")
    np_random = np.random.RandomState(0)
    
    # Check that the perceptron weights are initialized to a single vector with `dimensions` entries.
    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        p_weights = p.get_weights()

        number_of_parameters = 0

        for param in p.parameters():
            number_of_parameters += 1
            verify_node(param, 'parameter', (1, dimensions), 'PerceptronModel.parameters()')

        assert number_of_parameters == 1, ('Perceptron Model should only have 1 parameter')

    # Check that run returns a Tensor, and that the score in the node is correct
    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        point = np_random.uniform(-10, 10, (1, dimensions))
        score = p.run(Tensor(point))
        verify_node(score, 'tensor', (1,), "PerceptronModel.run()")
        calculated_score = score.item()
        
        # Compare run output to actual value
        for param in p.parameters():
            expected_score = float(np.dot(point.flatten(), param.detach().numpy().flatten()))

        assert np.isclose(calculated_score, expected_score), (
            "The score computed by PerceptronModel.run() ({:.4f}) does not match the expected score ({:.4f})".format(
            calculated_score, expected_score))

    # Check that get_prediction returns the correct values, including the
    # case when a point lies exactly on the decision boundary
    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        random_point = np_random.uniform(-10, 10, (1, dimensions))
        for point in (random_point, np.zeros_like(random_point)):
            prediction = p.get_prediction(Tensor(point))
            assert prediction == 1 or prediction == -1, (
                "PerceptronModel.get_prediction() should return 1 or -1, not {}".format(
                prediction))

            expected_prediction = np.where(np.dot(point, p.get_weights().data.T) >= 0, 1, -1).item()
            assert prediction == expected_prediction, (
                "PerceptronModel.get_prediction() returned {}; expected {}".format(
                    prediction, expected_prediction))

    tracker.add_points(2) # Partial credit for passing sanity checks

    print("Sanity checking perceptron weight updates...")

    # Test weight updates. This involves constructing a dataset that
    # requires 0 or 1 updates before convergence, and testing that weight
    # values change as expected. Note that (multiplier < -1 or multiplier > 1)
    # must be true for the testing code to be correct.
    dimensions = 2
    for multiplier in (-5, -2, 2, 5):
        p = models.PerceptronModel(dimensions)
        orig_weights = p.get_weights().data.reshape((1, dimensions)).detach().numpy().copy()
        if np.abs(orig_weights).sum() == 0.0:
            # This autograder test doesn't work when weights are exactly zero
            continue
        
        point = multiplier * orig_weights

        sanity_dataset = backend.Custom_Dataset(
            x=np.tile(point, (500, 1)),
            y=np.ones((500, 1)) * -1.0
        )
        
        p.train(sanity_dataset)
        new_weights = p.get_weights().data.reshape((1, dimensions)).detach().numpy()

        if multiplier < 0:
            expected_weights = orig_weights
        else:
            expected_weights = orig_weights - point

        if not np.all(new_weights == expected_weights):
            print()
            print("Initial perceptron weights were: [{:.4f}, {:.4f}]".format(
                orig_weights[0,0], orig_weights[0,1]))
            print("All data points in the dataset were identical and had:")
            print("    x = [{:.4f}, {:.4f}]".format(
                point[0,0], point[0,1]))
            print("    y = -1")
            print("Your trained weights were: [{:.4f}, {:.4f}]".format(
                new_weights[0,0], new_weights[0,1]))
            print("Expected weights after training: [{:.4f}, {:.4f}]".format(
                expected_weights[0,0], expected_weights[0,1]))
            print()
            assert False, "Weight update sanity check failed"

    print("Sanity checking complete. Now training perceptron")
    model = models.PerceptronModel(3)
    dataset = backend.PerceptronDataset(model)

    model.train(dataset)
    backend.maybe_sleep_and_close(1)

    assert dataset.epoch != 0, "Perceptron code never iterated over the training data"

    accuracy = np.mean(np.where(np.dot(dataset.x, model.get_weights().data.T) >= 0.0, 1.0, -1.0) == dataset.y)
    if accuracy < 1.0:
        print("The weights learned by your perceptron correctly classified {:.2%} of training examples".format(accuracy))
        print("To receive full points for this question, your perceptron must converge to 100% accuracy")
        return

    tracker.add_points(4)

@test('q2', points=6)
def check_regression(tracker):
    import models
    model = models.RegressionModel()
    dataset = backend.RegressionDataset(model=model)
    detected_parameters = None

    for batch_size in (1, 2, 4):
        inp_x = torch.tensor(dataset.x[:batch_size], dtype=torch.float, requires_grad=True)
        inp_y = torch.tensor(dataset.y[:batch_size], dtype=torch.float, requires_grad=True) 

        loss =  model.get_loss(inp_x, inp_y)

        verify_node(loss, 'tensor', (1,), "RegressionModel.get_loss()")
    

        grad_y = torch.autograd.grad(loss, inp_x, allow_unused=True, retain_graph=True)
        grad_x = torch.autograd.grad(loss, inp_y, allow_unused=True, retain_graph=True)
        
        assert grad_x[0] != None, "Node returned from RegressionModel.get_loss() does not depend on the provided input (x)"
        assert grad_y[0] != None, "Node returned from RegressionModel.get_loss() does not depend on the provided labels (y)"



    tracker.add_points(2) # Partial credit for passing sanity checks

    model.train(dataset)
    backend.maybe_sleep_and_close(1)

    data_x = torch.tensor(dataset.x,dtype=torch.float32)
    labels = torch.tensor(dataset.y, dtype=torch.float32)
    train_loss = model.get_loss(data_x, labels)
    verify_node(train_loss, 'tensor', (1,), "RegressionModel.get_loss()")
    train_loss = train_loss.item()

    # Re-compute the loss ourselves: otherwise get_loss() could be hard-coded
    # to always return zero
    train_predicted = model(data_x)

    verify_node(train_predicted, 'tensor', (dataset.x.shape[0], 1), "RegressionModel()")
    error = labels - train_predicted
    sanity_loss = torch.mean((error.detach())**2)

    assert np.isclose(train_loss, sanity_loss), (
        "RegressionModel.get_loss() returned a loss of {:.4f}, "
        "but the autograder computed a loss of {:.4f} "
        "based on the output of RegressionModel()".format(
            train_loss, sanity_loss))

    loss_threshold = 0.02
    
    if train_loss <= loss_threshold:
        print("Your final loss is: {:f}".format(train_loss))
        tracker.add_points(4)
    else:
        print("Your final loss ({:f}) must be no more than {:.4f} to receive full points for this question".format(train_loss, loss_threshold))

@test('q3', points=6)
def check_digit_classification(tracker):
    import models
    model = models.DigitClassificationModel()
    dataset = backend.DigitClassificationDataset(model)

    detected_parameters = None
    
    for batch_size in (1, 2, 4):
        inp_x = torch.tensor(dataset.x[:batch_size], dtype=torch.float, requires_grad=True)
        inp_y = torch.tensor(dataset.y[:batch_size], dtype=torch.float, requires_grad=True) 

        loss =  model.get_loss(inp_x, inp_y)

        verify_node(loss, 'tensor', (1,), "DigitClassificationModel.run()")

   
        grad_y = torch.autograd.grad(loss, inp_x, allow_unused=True, retain_graph=True)
        grad_x = torch.autograd.grad(loss, inp_y, allow_unused=True, retain_graph=True)
        
        assert grad_x[0] != None, "Node returned from RegressionModel.get_loss() does not depend on the provided input (x)"
        assert grad_y[0] != None, "Node returned from RegressionModel.get_loss() does not depend on the provided labels (y)"


    tracker.add_points(2) # Partial credit for passing sanity checks

    model.train(dataset)


    test_logits = model.run(torch.tensor(dataset.test_images)).data
    test_predicted = np.argmax(test_logits, axis=1).detach().numpy()
    test_accuracy = np.mean(test_predicted == dataset.test_labels)

    accuracy_threshold = 0.97
    if test_accuracy >= accuracy_threshold:
        print("Your final test set accuracy is: {:%}".format(test_accuracy))
        tracker.add_points(4)
    else:
        print("Your final test set accuracy ({:%}) must be at least {:.0%} to receive full points for this question".format(test_accuracy, accuracy_threshold))

@test('q4', points=7)
def check_lang_id(tracker):
    import models
    model = models.LanguageIDModel()
    dataset = backend.LanguageIDDataset(model)

    detected_parameters = None
    for batch_size, word_length in ((1, 1), (2, 1), (2, 6), (4, 8)):
        start = dataset.dev_buckets[-1, 0]
        end = start + batch_size
        inp_xs, inp_y = dataset._encode(dataset.dev_x[start:end], dataset.dev_y[start:end])
        inp_xs = torch.tensor(inp_xs[:word_length], requires_grad=True)

        output_node = model.run(inp_xs)
        verify_node(output_node, 'tensor', (batch_size, len(dataset.language_names)), "LanguageIDModel.run()")

        grad = torch.autograd.grad(torch.sum(output_node), inp_xs, allow_unused=True, retain_graph=True)
        for gradient in grad:
            assert gradient != None, "Output returned from LanguageIDModel.run() does not depend on all of the provided inputs (xs)"

        # Word length 1 does not use parameters related to transferring the
        # hidden state across timesteps, so initial parameter detection is only
        # run for longer words



    for batch_size, word_length in ((1, 1), (2, 1), (2, 6), (4, 8)):
        start = dataset.dev_buckets[-1, 0]
        end = start + batch_size
        inp_xs, inp_y = dataset._encode(dataset.dev_x[start:end], dataset.dev_y[start:end])
        inp_xs = torch.tensor(inp_xs[:word_length], requires_grad=True)
        loss_node = model.get_loss(inp_xs, inp_y)
        grad = torch.autograd.grad(loss_node, inp_xs, allow_unused=True, retain_graph=True)
        for gradient in grad:
            assert gradient != None, "Output returned from LanguageIDModel.run() does not depend on all of the provided inputs (xs)"


    tracker.add_points(2) # Partial credit for passing sanity checks

    model.train(dataset)


    accuracy_threshold = 0.81
    test_accuracy = dataset.get_validation_accuracy()
    if test_accuracy >= accuracy_threshold:
        print("Your final test set accuracy is: {:%}".format(test_accuracy))
        tracker.add_points(5)
    else:
        print("Your final test set accuracy ({:%}) must be at least {:.0%} to receive full points for this question".format(test_accuracy, accuracy_threshold))

@test('q5', points=4)
def check_convolution(tracker):
    import models

    model = models.DigitConvolutionalModel()
    dataset = backend.DigitClassificationDataset2(model)

    def conv2d(a, f):
        s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
        strd = np.lib.stride_tricks.as_strided
        subM = strd(a, shape = s, strides = a.strides * 2)
        return np.einsum('ij,ijkl->kl', f, subM)

    detected_parameters = None
    
    for batch_size in (1, 2, 4):
        inp_x = torch.tensor(dataset[:batch_size]['x'], dtype=torch.float, requires_grad=True)
        inp_y = torch.tensor(dataset[:batch_size]['label'], dtype=torch.float, requires_grad=True) 
        loss =  model.get_loss(inp_x, inp_y)

        verify_node(loss, 'tensor', (1,), "DigitClassificationModel.run()")

   
        grad_y = torch.autograd.grad(loss, inp_x, allow_unused=True, retain_graph=True)
        grad_x = torch.autograd.grad(loss, inp_y, allow_unused=True, retain_graph=True)
        
        assert grad_x[0] != None, "Node returned from RegressionModel.get_loss() does not depend on the provided input (x)"
        assert grad_y[0] != None, "Node returned from RegressionModel.get_loss() does not depend on the provided labels (y)"
    
    for matrix_size in (2, 4, 6): #Test 3 random convolutions to test convolve() function
        weights = np.random.rand(2,2)
        input = np.random.rand(matrix_size, matrix_size)
        student_output = models.Convolve(torch.Tensor(input), torch.Tensor(weights))
        actual_output = conv2d(input,weights)
        assert np.isclose(student_output, actual_output).all(), "The convolution returned by Convolve() does not match expected output"

    tracker.add_points(1) # Partial credit for testing whether convolution function works

    model.train(dataset)


    test_logits = model.run(torch.tensor(dataset.test_images)).data
    test_predicted = np.argmax(test_logits, axis=1).detach().numpy()
    test_accuracy = np.mean(test_predicted == dataset.test_labels)

    accuracy_threshold = 0.80
    if test_accuracy >= accuracy_threshold:
        print("Your final test set accuracy is: {:%}".format(test_accuracy))
        tracker.add_points(3)
    else:
        print("Your final test set accuracy ({:%}) must be at least {:.0%} to receive full points for this question".format(test_accuracy, accuracy_threshold))

@test('q6', points=1)
def check_attention(tracker):
    import models
    
    for block_size in [2,4,16]:
        layer_size = np.random.randint(2,10)

        att_block = models.Attention(layer_size, block_size)
        batch_size = np.random.randint(1,10)

        input = torch.rand(batch_size, block_size, layer_size)

        
        #Define weights to manually set k, q, and v layers to
        k_weight = torch.rand(layer_size,layer_size).reshape(layer_size,layer_size)
        q_weight = torch.rand(layer_size,layer_size).reshape(layer_size,layer_size)
        v_weight = torch.rand(layer_size,layer_size).reshape(layer_size,layer_size)

        #Manually assign attention weights, makes autograding easier
        with torch.no_grad():
            att_block.k_layer.weight = torch.nn.Parameter(torch.ones((layer_size,layer_size)) * k_weight)
            att_block.q_layer.weight = torch.nn.Parameter(torch.ones((layer_size,layer_size)).reshape(layer_size,layer_size) * q_weight)
            att_block.v_layer.weight = torch.nn.Parameter(torch.ones((layer_size,layer_size)).reshape(layer_size,layer_size) * v_weight)
        
        
        T = input.shape[1]
        
        expected_output = torch.matmul(att_block.k_layer(input),torch.movedim(att_block.q_layer(input),1,2))/layer_size**0.5

        expected_output = expected_output.masked_fill(att_block.mask[:,:,:T,:T] == 0, float('-inf'))[0]
        
        output = torch.matmul(nn.functional.softmax(expected_output, dim=-1),att_block.v_layer(input))


        assert torch.all(torch.isclose(output, att_block(input))), "The output returned by Attention() does not match expected output"
    tracker.add_points(1)
if __name__ == '__main__':
    main()
