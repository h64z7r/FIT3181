java c
FIT3181:   Deep   Learning   (2024)
Deep   Neural   Networks
Due:   11:55pm Sunday, 8 September 2024 (Sunday)
Important   note: This   is an   individual assignment.   It contributes 25% to your final   mark.   Read the   assignment   instructions   carefully.
What   to   submit
This assignment   is to   be completed   individually and submitted to   Moodle unit   site.   By   the   due   date, you   are   required   to   submit   one   single   zip   file,   named      xxx_assignment01_solution.zip where    xxx   is your student   ID, to the corresponding Assignment   (Dropbox)   in   Moodle. You can   use   Google   Colab   to   do   Assigmnent   1   but   you   need   to   save   it   to   an         *.ipynb   file   to   submit   to   the   unit   Moodle.
More   importantly,   if you   use Google Colab to do this assignment, you   need to first   make   a   copy   of this   notebook   on your   Google   drive   .
For example, if   your   student ID is   12356,   then gather all of   your assignment solution   to folder, create a zip file named   123456_assignment01_solution.zip   and   submit   this   file.
Within this zipfolder, you must submit the following files:
1. Assignment01_solution.ipynb: this is your Python   notebook   solution   source file.
2. Assignment01_output.html: this is the output of your Python   notebook solution   exported   in   html format.
3. Any extra files or folder needed to complete your assignment   (e.g.,   images   used   in   your   answers).
Since the   notebook   is quite   big to load and work together, one   recommended   option   is   to   split   solution   into   three   parts   and work   on   them   seperately.   In   that   case, replace Assignment01_solution.ipynb   by three notebooks: Assignment01_Part1_solution.ipynb, Assignment01_Part2_solution.ipynb   and   Assignment01_Part3_solution.ipynb
You can   run your codes on Google Colab.   In this   case, you   have   to   make   a   copy   of your   Google   colab   notebook   including the   traces   and   progresses   of model training before   submitting.
Part   1: Theory   and   Knowledge   Questions          [Total   marks   for   this   part: 30   points]The first   part of this assignment   is to demonstrate your knowledge   in deep   learning that   you   have   acquired   from   the   lectures   and   tutorials   materials.   Most   of the   contents   in this assignment are drawn from the   lectures and tutorials from weeks   1   to   4.   Going   through   these   materials   before   attempting   this   part   is   highly         recommended.
Question   1.1 Activation function   plays an   important   role   in modern   Deep   NNs.   For each of the activation functions   below, state   its   output   range, find   its derivative (show your steps),   and   plot the activation fuction   and   its   derivative
(b) Gaussian   Error   Linear   Unit   (GELU):   GELU(x) =   xΦ(x)   where   Φ(x) is   the      probability   cummulative   function   of   the   standard   Gaussian   distribution   or    Φ(x) =   P (X   ≤ x) where   X   ~ N (0, 1)   .   In   addition, the   GELU   activation   fuction   (the   link   for   the   main   paper   (https://arxiv.org/pdf/1606.08415v5.pdf))   has
been   widely   used   in   the   state-of-the-art   Vision   for   Transformers   (e.g.,   here   is   the   link   for   the   main   ViT   paper   (https://arxiv.org/pdf/2010.11929v2.pdf)).    [1.5   points]
Write your answer   here.   You can add more cells if   needed.
Question   1.2: Assume that we feed a data   point x   with a   ground-truth   label   y = 2   to the   feed-forward   neural   network   with the      ReLU   activation   function as shown   in the following figure   
(a) What   is   the   numerical   value   of   the   latent   presentation   h1 (x)?    [1   point]
(b) What   is   the   numerical   value   of   the   latent   presentation   h2   (x)?       [1   point]
(c) What   is   the   numerical   value   of   the   logith3   (x)?       [1   point]
(d) What   is   the   corresonding   prediction   probabilities   p(x)?       [1   point]
(e) What   is   the   predicted   label   y(^)?   Is   it   a   correct   and   an   incorect   prediction?   Remind   that   y   = 2. [1   point]
(f) What   is   the   cross-entropy   loss   caused   by   the   feed-forward   neural   network   at   (x,   y)?   Remind   that   y   = 2.    [1   point](g) Why   is   the   cross-entropy   loss   caused   by   the   feed-forward   neural   network   at   (x,   y) (i.e.,   CE(1y,   p(x)))   always   non-negative?   When   does   this   CE(1y,   p(x))   loss get the value 0?   Note that you   need to answer this question for   a   general   pair   (x,   y) and   a   general feed-forward   neural   network   with,   for   example   M   = 4      classes?       [1   point]
You must show both formulas and numerical results for earning full mark. Although it is optional, it is great if   you show   your PyTorch code for   your computation.
Question   1.3:
For   Question   1.3, you   have   two   options:
·         (1) perform   the   forward, backward   propagation, and   SGD   update   for      one   mini-batch   (10   points),   or
·         (2) manually implement a feed-forward neural network that can work on real tabular datasets (20   points).
You   can   choose   either   (1)   or   (2)   to   proceed.
Option   1                   [Total   marks   for   this   option:   10   points]
Assume that we are constructing a multilayered feed-forward   neural   network for a   classification   problem with three   classes where   the   model   parameters   will   be   generated   randomly   using   your   student   ID. The   architecture   of   this   network   is   3(Input)   → 5(ELU)   → 3(output) as   shown   in the following figure.   Note that the   ELU   has the same formula as the one   in   Q1.1.
We feed a batch X   with the labels   Y as   shown   in   the   figure. Answer the following   questions.   
You   need to show   both formulas,   numerical   results, and your PyTorch code   for your computation   for   earning   full   marks.
In      [      ]:
Out[3]:

In      [      ]:
#Code   to   generate   random   matrices   and   biases   for   W1,   b1,   W2,   b2
Forward   propagation
(a) What   is   the   value   of   h(¯)1 (x) (the   pre-activation   values   of   h1   )?    [0.5   point]
In      [      ]:
(b) What   is   the   value   of   h1 (x)?       [0.5   point]
In      [      ]:
(c) What   is   t代 写FIT3181: Deep Learning (2024)Python
代做程序编程语言he   predicted   value   y(^)?    [0.5   point]
In      [      ]:
(d) Suppose that we use the cross-entropy   (CE)   loss. What   is the value   of the   CE   loss   l incurred   by the   mini-batch?   [0.5   point]
In      [      ]:
Backward   propagation
(e) What   are   the   derivatives         ,    , and   ?    [3   points]
In      [      ]:
(f) What   are   the   derivatives      ,    ,    ,   and      ?       [3   points]
In      [      ]:
SGD   update
(g) Assume   that   we   use   SGD   with   learning   rate   η   = 0.01   to   update   the   model   parameters. What   are   the   values   of   W   2   ,   b2   and   W   1 ,   b1    after   updating?    [2   points]
In      [      ]:
Option 2          [Total   marks   for   this   option:   20   points]
In      [      ]:
import   torch
from   torch.utils.data   import   DataLoader
from   torchvision   import   datasets,   transforms
In Option 2, you   need to   implement a feed-forward   NN manually   using   PyTorch and   auto-differentiation   of   PyTorch.   We   then   manually   train   the   model on the   MNIST dataset.
We first download the    MNIST   dataset   and   preprocess   it.
In      [      ]:
Each   data   point   has   dimension         [28,28]   . We   need   to   flatten   it   to   a   vector   to   input   to   our   FFN.
In      [      ]:
train_dataset.data   =   train_data.data.view(-1,   28*28)      test_dataset.data   =   test_data.data.view(-1,   28*28)
train_data,   train_labels   =   train_dataset.data,   train_dataset.targets      test_data,   test_labels   =   test_dataset.data,   test_dataset.targets
print(train_data.shape,   train_labels.shape)
print(test_data.shape,   test_labels.shape)
In      [      ]:
train_loader   =   DataLoader(dataset=train_dataset,   batch_size=64,   shuffle=True)      test_loader   =   DataLoader(dataset=test_dataset,   batch_size=64,   shuffle=False)
Develop the feed-forward   neural   networks
(a) You   need to develop the class    MyLinear   with the following   skeleton. You   need   to   declare   the weight   matrix   and   bias   of this   linear   layer.    [3   points]
In      [      ]:
(b) You   need to develop the class    MyFFN   with the following   skeleton       [7   points]
In      [      ]:
In      [      ]:
myFFN   =   MyFFN(input_size   =   28*28,   num_classes   =   10,   hidden_sizes   =   [100,   100],   act   =   torch.nn.ReLU)      myFFN.create_FFN()
print(myFFN)
(c) Write the code to evaluate the accuracy of the current      myFFN   model   on   a   data   loader   (e.g.,   train_loader   or test_loader).       [2.5   points]
In      [      ]:
(c) Write the code to evaluate the loss of the   current      myFFN   model   on   a   data   loader   (e.g., train_loader   or test_loader).    [2.5   points]
In      [      ]:
def   compute_loss(model,   data_loader):
"""
This   function   computes   the   loss   of   the   model   on   a   data   loader
"""
#Your   code   here
Train on the    MNIST   data with   50   epochs   using      updateSGD   .
In      [      ]:
(d)   Implement the function    updateSGDMomentum   in the class and   train   the   model with   this   optimizer   in       50   epochs. You can update   the   corresponding function   in the      MyFNN   class.       [2.5   points]
In      [      ]:
(e)   Implement the function    updateAdagrad   in the class and   train   the   model with   this   optimizer   in       50   epochs. You can update   the   corresponding function   in   the   MyFNN   class.    [2.5   points]
In      [      ]:
Part 2:   Deep   Neural   Networks (DNN)       [Total   marks   for   this   part: 25   points]
The second   part of this assignment   is to demonstrate your basis   knowledge   in deep   learning   that   you   have   acquired   from   the   lectures   and   tutorials   materials.   Most of the contents in this assignment are drawn from   the   tutorials   covered   from weeks   1   to   2.   Going   through   these   materials   before   attempting   this assignment is   highly   recommended.In the second   part of this assignment, you are going to work with the   FashionMNIST   dataset for   image   recognition   task.   It   has   the   exact   same   format   as   MNIST   (70,000 grayscale   images   of   28   × 28   pixels   each   with   10 classes),   but   the   images   represent   fashion   items   rather   than   handwritten   digits,   so   each   class   is   more    diverse, and the problem is significantly   more   challenging   than   MNIST.
In      [      ]:
import   torch
from   torch.utils.data   import   DataLoader
from   torchvision   import   datasets,   transforms   torch.manual_seed(1234)
Load the Fashion MNIST   using       torchvision
In      [      ]:
torch.Size([60000,   28,   28])   torch.Size([60000])   torch.Size([10000,   28,   28])   torch.Size([10000])   torch.Size([60000,   784])   torch.Size([60000])
torch.Size([10000,   784])   torch.Size([10000])Number   of   training   samples:   18827      Number   of   training   samples:   16944      Number   of   validation   samples:   1883
Question 2.1: Write the code to visualize a mini-batch in      train_loader   including   its   images   and   labels.      [5   points]
In      [      ]:
####Question   2.2: Write   the   code   for   the   feed-forward   neural   net   using   PyTorch       [5   points]
We   now   develop   a   feed-forward   neural   network   with   the   architecture   784   → 40(ReLU)   → 30(ReLU)   →   10(softmax)   . You   can   choose   your   own   way   to implement your network and an optimizer of interest. You   should   train   model   in   50 epochs   and   evaluate   the   trained   model   on   the   test   set.
In      [      ]:
Question 2.3: Tuning   hyper-parameters with grid search       [5   points]
Assume   that   you   need   to   tune   the   number   of   neurons   on   the   first   and   second   hidden   layers   n1       ∈   {20,   40}   , n2      ∈   {20,   40}   and   the   used   activation   function act   ∈   {sigmoid,   tanh,   relu}   . The   network   has the architecture   pattern 784   → n1   (act)   → n2 (act)   →   10(softmax) where   n1   ,   n2 ,   and   act are   in   their
grides. Write the code to tune the   hyper-parameters n1   ,   n2 , and   act.   Note   that   you   can freely   choose   the   optimizer   and   learning   rate   of   interest   for   this   task.
In      [      ]:

   

         
加QQ：99515681  WX：codinghelp
