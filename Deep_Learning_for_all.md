# chap01 - Tensor Manipulation 

```python
a: Tensor
b: Tensor

# 아래 둘은 동일하다
a.matmul(b)
a@b

# 아래 둘은 동일하다
a.mul(b)
a*b

# 아래 둘은 동일하다
a.mul_(b)
a = a.mul(b)


# dimention 1 방향으로는 확장 가능
# dimention 0 방향으로는 확장 불가능

# a = [[1, 2, 3], [4, 5, 6]]
# b = [1, 2]
# a * b 불가능

# a = [[1, 2, 3], [4, 5, 6]]
# b = [[2], [3]]
# a * b = [[2, 4, 6], [12, 15, 18]] 가능
```


# chap02 - Linear regression

```python
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.01)  # Stochestic Gradient descent
nb_epochs = 1000

for epoch in range(1, nb_epochs + 1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    optimizer.zero_grad()   # 이 명령어가 없으면 이전 loop의 결과에 영향을 받게됨
    cost.backward()
    optimizer.step()    # step 1회당 parameters update 1회

```

# chap03 - Deeper Look at Gradient Descent

* bias = 0
* Gradient descent 직접 구현

```python
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# 모델 초기화
W = torch.zeros(1)
# learning rate 설정
lr = 0.1
nb_epochs = 10
for epoch in range(nb_epochs + 1):
 
    # H(x) 계산
    hypothesis = x_train * W
    
    # cost gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), cost.item()
    ))
    # cost gradient로 H(x) 개선
    W -= lr * gradient
```

# chap04-1 - Multivariate Linear Regression

```python
# 모델
# class(super class) : 상속
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 98, 100],
    [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
# W = torch.zeros((3, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
model = MultivariateLinearRegressionModel()

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    # hypothesis = x_train.matmul(W) + b # or .mm or @
    Hypothesis = model(x_train)

    # cost 계산
    # cost = torch.mean((hypothesis - y_train) ** 2)
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), 
        cost.item()
    ))
```

# chap04-2 - Loading Data
* Minibatch Gradient Descent

```python

from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self):
    self.x_data = [[73, 80, 75],
        [93, 88, 93],
        [89, 91, 90],
        [96, 98, 100],
        [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()

from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    )

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
    ))
```

# chap05 - Logistic Regression

```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 100
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))
```

# chap06 - Softmax Classification

## Discrete probability distribution
* Uniform distrinution(정규분포 등)에서는 확룰이 면적이다
    - p(X = x)를 구할 수 없음
* Discrete probability distribution는 p(X=x)를 구할 수 있음
    - ex) 주사위를 굴렸을 때 6이 나올 확률
        + p(x=6) = $\frac{1}{6}$

## Softmax
* 합이 1이 되는 확률

## Cross Entropy
* 두개의 확률분포가 주어졌을 때 둘이 얼마나 비슷한지 나타낼 수 있는 수치

## Softmax Classification python code
```python
# Low level
torch.log(F.softmax(z, dim=1))
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
```

```python
# High level
F.log_softmax(z, dim=1)
F.nll_loss(F.log_softmax(z, dim=1), y)
```

```python
# use library
F.cross_entropy(z, y)
```

```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output이 3!

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

# chap07-1 Tips

## Learning Rate
* learning rate이 너무 크면 diverge 하면서 cost 가 점점 늘어난다 (overshooting)
* learning rate이 너무 작으면 cost가 거의 줄어들지 않는다
* 때문에 적정한 learning rate를 잘 설정해야한다

## Data preprocessiong
* 데이터를 zero-center하고 normalize하자
    - $ x'_j = \frac{x_j - \mu_j}{\sigma_j} $
        + $\mu$ : 표준편차
        + $\sigma$ : 평균


## Overfitting
* 방지법
    1. 더 많은 학습 데이터
    2. 더 적은 feature
    3. Regularization

# chap07-2 MNIST

```python
# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# MNIST data image of shape 28 * 28 = 784
linear = torch.nn.Linear(784, 10, bias=True).to(device)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

# Test the model using test sets
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
```


# chap08-1 - Perceptron

## XOR
```python
# Lab 9 XOR
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# nn layers
linear1 = torch.nn.Linear(2, 10, bias=True)
linear2 = torch.nn.Linear(10, 10, bias=True)
linear3 = torch.nn.Linear(10, 10, bias=True)
linear4 = torch.nn.Linear(10, 1, bias=True)
sigmoid = torch.nn.Sigmoid()

# model
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid, linear3, sigmoid, linear4, sigmoid).to(device)

# define cost/loss & optimizer
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)  # modified learning rate from 0.1 to 1

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    # cost/loss function
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())

# Accuracy computation
# True if hypothesis>0.5 else False
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(), '\nAccuracy: ', accuracy.item())
```

# chap08-2 - Multi Layer Perceptron

## Backpropagation
```python
X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)[:1000]
Y_test = mnist_test.test_labels.to(device)[:1000]
i = 0
while not i == 10000:
    for X, Y in data_loader:
        i += 1

        # forward
        X = X.view(-1, 28 * 28).to(device)
        Y = torch.zeros((batch_size, 10)).scatter_(1, Y.unsqueeze(1), 1).to(device)    # one-hot
        l1 = torch.add(torch.matmul(X, w1), b1)
        a1 = sigmoid(l1)
        l2 = torch.add(torch.matmul(a1, w2), b2)
        y_pred = sigmoid(l2)

        diff = y_pred - Y

        # Back prop (chain rule)
        d_l2 = diff * sigmoid_prime(l2)
        d_b2 = d_l2
        d_w2 = torch.matmul(torch.transpose(a1, 0, 1), d_l2)

        d_a1 = torch.matmul(d_l2, torch.transpose(w2, 0, 1))
        d_l1 = d_a1 * sigmoid_prime(l1)
        d_b1 = d_l1
        d_w1 = torch.matmul(torch.transpose(X, 0, 1), d_l1)

        w1 = w1 - learning_rate * d_w1
        b1 = b1 - learning_rate * torch.mean(d_b1, 0)
        w2 = w2 - learning_rate * d_w2
        b2 = b2 - learning_rate * torch.mean(d_b2, 0)

        if i % 1000 == 0:
            l1 = torch.add(torch.matmul(X_test, w1), b1)
            a1 = sigmoid(l1)
            l2 = torch.add(torch.matmul(a1, w2), b2)
            y_pred = sigmoid(l2)
            acct_mat = torch.argmax(y_pred, 1) == Y_test
            acct_res = acct_mat.sum()
            print(acct_res.item())

        if i == 10000:
            break
```

# chap09-1 - ReLU(활성화 함수)

* Backpropagation과정에서 sigmoid함수의 일부 부분에서는 0과 매우 가까운 미분값이 발생
    - 이에대한 gradient값이 사라지는 현상 발생(vanishing)
* ReLU
    - $f(x) = max(0, x)$
        + x가 음수일 때는 0, 아니면 x
            + x가 음수일 경우에 gradient값이 0이 되는 문제가 있지만 sigmoid보다는 잘 작동됨
* Obtimizer
    - 실제 동작을 그래프 상으로 표현 </br>
        ![http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html](http://2.bp.blogspot.com/-q6l20Vs4P_w/VPmIC7sEhnI/AAAAAAAACC4/g3UOUX2r_yA/s400/s25RsOr%2B-%2BImgur.gif)
    - Optimizer의 발전 과정 </br>
        ![https://blog.naver.com/PostView.nhn?blogId=another0430&logNo=222063836606](https://blog.kakaocdn.net/dn/bQ934t/btqASyVqeeD/ozNDSKWvAbxiJb7VtgLkSk/img.png) </br>
        

# chap09-2 - Weight initialization

* Weight initialization을 한 것이 안한 것보다 학습 성능이 좋다

## Restricted Boltzmann Machine(RBM)
* Restricted : 같은 layer 내의 Node에는 연결이 없지만, 이웃 layer의 Node에는 모두 연결됨
* Pre-training : 
    1. x와 hidden 1을 RBM으로 학습
        + RBM으로 학습 : input x로 y를 얻는다, y를 통해 x'을 복원한다
    2. x와 hidden 1 사이의 parameter 고정하고 hidden 2를 추가하여 RBM으로 학습
    3. $\dots$
    4. Fine-tuning
        + 입력 x와 출력 y를 통해 backpropagation으로 parameter 갱신

## Xavier / He initialization
* 무작위 값이 아닌 Layer마다의 특징을 가지고 초기화를 한다

### Xavier initialization
* Xavier Normal initialization
    - $W ~ N(0, Var(W))$
    - $Var(W) = \sqrt{\frac{2}{n_{in} + n_{out}}}$
        + $n_{in}$ : layer의 input 수
            + 이전 layer의 수
        + $n_{out}$ : layer의 output 수
            + 다음 layer의 수
* Xavier Uniform initialization
    - $W ~ U(-\sqrt{\frac{6}{n_{in} + n_{out}}}, +\sqrt{\frac{6}{n_{in} + n_{out}}})$
* 위의 수식을 이용해서 초기화 한다
* 비선형 함수(sigmoid, tanh 등)에서 성능이 뛰어나지만, ReLU 함수에서 사용시 출력값이 0으로 수렴하게되는 문제가 있다

### He initialization
* Xavier의 변형
* He Normal initialization
    - $W ~ N(0, Var(W))$
    - $Var(W) = \sqrt{\frac{2}{n_{in}}}$
        + $n_{in}$ : layer의 input 수
* He Uniform initialization
    - $W ~ U(-\sqrt{\frac{6}{n_{in}}}, +\sqrt{\frac{6}{n_{in}}})$
* Xavier에서 $n_{out}$ 을 제외한 형태
* 위의 수식을 이용해서 초기화 한다

* Xavier 구현
```python
def xavier_uniform_(tensor, gain=1):

    """
    .. math::
    a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}
    Also known as Glorot initialization.
    Args:
    tensor: an n-dimensional `torch.Tensor`
    gain: an optional scaling factor
    Examples:
    >>> w = torch.empty(3, 5)
    >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std # Calculate uniform bounds from standard deviation
    with torch.no_grad():
    return tensor.uniform_(-a, a)
```

* Xavier 적용

```python
# nn layers
linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)
relu = torch.nn.ReLU()

# xavier initialization
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)

# model
model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

# Test the model using test sets
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())
```


# chap09-3 Dropout

## Overfitting
* 방지법
    1. 더 많은 학습 데이터
    2. 더 적은 feature
    3. Regularization
    4. **Dropout**

## Dropout

* 학습을 진행하면서 각 layer 내의 Node를 설정된 비율에 따라 무작위로 활성화, 비활성화를 반복하는 것

```python
model.train()
# dropout 적용
# 학습시에 호출

model.eval()
# dropout 미적용
# test시에 호출
```


# chap09-4 Batch Normalization

## Gradient Vanishing / Exploding
* Gradient Vanishing
    - gradient값이 0과 가까워지면서 loss에 미치는 영향이 사라짐
* Gradient Exploding
    - gradient 값이 너무 큰 값이 나오거나 NAND값이 나오는 경우

## Solution
* activation function 변경
    - Sigmoid -> ReLU
* Initialization을 잘 해보기
    - RBM, Xavier, He initialization
* Learning rate
    - Gradient Exploding 해결책

## Internal Covariate Shift
* train set과 test set의 분포가 차이가 있고, 이것 때문에 문제들이 발생한다
    - 입력과 출력의 분포가 다르다
* Gradient Vanishing / Exploding을 발생시키는 원인

## Batch Normalization
* 전체 학습 과정이 안정적이된다
* 학습속도의 가속 등의 이점이 있음
* 각 layer마다 normalization하는 layer를 배치한다
* mini-batch마다 normalization을 수행한다

* $\displaystyle \hat{x_i} := \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
    - $\epsilon$ : 분모가 0이 되는 것을 방지
* $y_i := \gamma \hat{x_i} + \beta$
    - scale, shift 수행
    - $\gamma, \beta$ : backpropagation을 통해 학습한 값
* model.train()을 사용하면 각각의 경우에 따라 일부 노드는 Dropout이 되기 때문에 매번 다른 값의 평균과 표준편차를 얻게 된다
* sample mean과 sample variance를 얻었을 때 이를 저장해놓고 test를 할 때는 이를 가져와서 사용한다


# chap10-1 Convolution

## Convolution
* Filter(kernel)를 stride값 만큼 이동시키면서 filter에 겹처진 부분의 각 원소를 곱해서 모두 더한 값을 출력으로 하는 연산
    - stribe
        + filter를 한번에 얼마나 이동할 것인가
    - padding(zero-padding)
        + input의 주변을 0으로 주변을 얼마나 감쌀 것인가

### nn.conv2d

* $\displaystyle output = \frac{input size - filter size + 2*padding}{stride} + 1$

* ex1)
    - input image size : 227 $\times$ 227
    - filter size : 11 $\times$ 11
    - stride : 4
    - padding : 0
    - output image size : 55

* ex2)
    - input image size : 64 $\times$ 64
    - filter size : 7 $\times$ 7
    - stride : 2
    - padding : 0
    - output image size : 28.5 -> 28

* ex3)
    - input image size : 32 $\times$ 32
    - filter size : 5 $\times$ 5
    - stride : 1
    - padding : 2
    - output image size : 32

### Pooling
+ Max puling
    - Kernel 내의 최대값을 반환
    - torch.nn.MaxPool2d
+ Average puling 


# chap11 RNN

## RNN
* Sequential data를 다루기 위해 만들어짐
    - ex) 강아지와 고양이를 구분하는 문제를 만들었을 때 두 사진이 입력되는 순서가 바뀌어도 결과가 바뀌지 않음 : Sequential 하지 않음
    - 단어, 문장, 시계열 데이터 등이 sequential data에 해당한다
* RNN이전의 sequential data를 다룬 방법
    - Data에 추가적으로 potition index를 추가함

### RNN의 구조
* t번째 입력이 셀로 들어가면 t번째 출력이 나오고, hidden state가 다음 셀로 입력됨
    - 이전 입력의 결과가 다음 결과에 영향을 미침
* hello 등의 경우 어떤 l의 경우에는 l이 다음으로 출력해야하고 어떤 경우에는 o를 출력해야하기 때문에 sequential data를 다루기 어렵다
    - 이전 입력값의 정보를 hidden state를 통해 얻을 수 있어서 sequential data를 잘 처리한다
    - RNN은 모든 셀이 전부 parameter를 공유한다
        + 긴 sequence에 대해서도 잘 처리할 수 있다
* 복잡한 cell을 사용할 수록 학습 정도(tunability)가 낮아진다
    - 복잡한 cell을 사용하면 같은 학습 수준에서 더 나은 성능을 보이지만, 이 학습 수준까지 학습하는데 더 오래 걸린다
    - RNN < GRU < LSTM 순으로 정점 복잡한 셀을 사용한다

### cell의 동작
* $h_t = f(h_{t-1}, x_t)$
* $h_t = tanh(W_hh_{t-1} + W_xx_t)$

### RNN의 예시
* LSTM, GRU 등

## Seq2seq

* 문장에 대한 답변은 뒷부분에서 달라질 수 있기 때문에 끝까지 들어봐야한다

### Encoder-Decoder

* Encoder
    - 입력된 sequence를 vector의 형태로 압축하고 decoder로 전달함
* Decoder
    - Encoder에서 넘겨준 vecotr를 첫 셀의 hiddne state로 사용하고, start flag와 함께 모델을 시작함
    - 첫 셀에서의 output을 문장의 시작으로 두고, 이를 두번째 셀의 입력으로 사용됨
