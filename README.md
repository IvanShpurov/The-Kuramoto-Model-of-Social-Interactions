# The Kuramoto Model of Social Interactions
### *This is code for the paper "Emergence of the Neural Synchrony in the Multi-Agent simulation" presented at AROBS 2022 conference*.
# Table of contents
- [Why I did this](#introduction)
- [Two oscillators](#Two_oscillators)
- [Kuramoto Model 101](#paragraph1)
  - [Order parameter](#subparagraph1)
- [Usage](#paragraph2) 
  - [Basic Model](#Basic_model)
  - [Stochastic rewiring](#Stochastic)
  - [Contionous change](#Continous)
- [Sources](#Sources)

## Why I did this <a name="introduction"></a>
*Here comes somewhat lengthy introduction which explains the rationale of my project and also supplies the reader with a number of relatively useless facts. If all you need is the plain implementation of the Kuramoto model in Python, skip to [here](#Basic_model), if you are unsure why you are on this page continue reading.* 

I always found synchrony quite fascinating. And I was in good company - as far back as 17-th century renowned Dutch mathematician [Christiaan Huygens](https://en.wikipedia.org/wiki/Christiaan_Huygens) was perplexed by the uncanny capacity of pendulum clocks to synchronize their swings. The motion of the two pendula was such that their periods were identical but their displacements were opposite in direction. Being a scientist he rejected the out-of-the-box explanation his age had to offer (clearly, devils’ work, consider consulting a priest or sprinkling your setup with holy water) and his inquiries into the topic gave birth to a whole new branch of physics. 

After numerous rearrangements to the setup, he figured out that the pendulum clocks can in fact interact, being both attached to the heavy navy beam, which held them in place. Vibrations caused by the pendulum movements could traverse the beam back and forth - this is known as weak coupling. Synchronization of this sort is quite ubiquitous in nature, yet models which describe it prove themselves to be especially hard to tackle, as one has to keep in mind too many changing variables - phase, amplitude *etcetera*. Not, that it is in principle impossible - but challenging, and you are not going to find any of this stuff here. Luckily for us, there is a considerably simpler model, known as the [phase model](http://www.scholarpedia.org/article/Phase_model), which omits a lot of complexity, while preserving the crux of the problem. This model could be used to describe the behavior of [2](#Two_oscillators) or [more](#paragraph1) weakly coupled oscillators.

There is though somewhat less known kind of synchrony I’m interested in. It is known as interpersonal synchrony. We all are exposed to this phenome during our daily routine, although usually, we don’t think much of it. When two peoples walk side by side, they would adjust their stride length and pace to accommodate each other. This is behavioral synchrony. Its existence comes hardly a surprise - what is however a surprise, in my opinion, is that the degree of synchronization correlates with mutual affiliation between individuals. Couples walking together do so more synchronously than strangers. The outcome of marital argument can be inferred from the mutual behavioral synchronization between the spouses (that experiment is weird I agree).

Apart from the aforementioned behavioral synchrony there exist neural and physiological synchrony. This term refers, respectively to synchronization between neural time-series (EEG and fMRI recordings) and recordings of physiological variables (Heart rate, Breathing, many others). And again in the experimental setting well-synchronized participants rewarded each other with higher social ratings, had a higher proclivity to cooperation, and performed better when they had to work together.

Now, I was careful to use the word “correlation” as it does not imply causality. The scientific community is still in debate if the propensity to synchronize enables the social cohesions between individuals or is itself resultant from social interactions. Personally, I think that there is significant evidence that synchrony might play a causal role - thus, artificially increasing synchronization between the participants, using transcranial magnetic stimulation (putting somebody's head inside a big magnetic coil to induce electric currents in a specific area of the brain. Check [this](#https://www.youtube.com/watch?v=cguWM9PGMhE&ab_channel=NeurosoftRussia), it is quite hilarious to observe how a guy forgets his mother tong) makes their work as a pair more efficient. Applying the same protocol to rats, using implanted electrodes makes synchronized animals friends. Curiously enough, you could not destroy existing rat friendship using out-of-phase stimulation. 

Apart from the experiments I briefly reviewed (and there are many more, check the references) there are computational experiments, which used versions of the Kuramoto model to explore the emergence of synchrony between artificial agents. However, they focused on dyadic interactions between two participants. I wondered how such dynamics would affect the group formation. 

I’ll sort of try to outline my thinking here (Thank god I don’t need to pretend to be a serious scientist while doing so, as I did in the actual paper). 

Let's assume that we are somewhat more likely to interact with people we are in sync with and we could infer how in sync we are with someone from their speech pattern, movement tempo, and other perceivable cues. Now imagine a room full of strangers who are just starting to establish their network of social connections. That is a very complicated way to say “imagine a cocktail party”, I know. Let’s say that peoples are more likely to strike a conversation with someone they feel they are in sync with. While these social interactions last, individuals get more and more synchronized. So we have a positive feedback loop - **interacting with someone causes your internal rhythms to synchronize, yet you base your initial choice of conversational partners on the preexisting degree of synchrony**. Such process results in the “rich get richer” phenomenon - if you try the [simulation model](#Stochastic) for yourself, you will see that individuals end up with a very different number of friends and the whole population get’s segregated into groups. Sort of similar to a real party, right? 




## Two oscillators <a name="Two_oscillators"></a>
Check the Two_coupled_oscilators.ipynb notebook for a simple introduction to coupled oscilators. 
## Kuramoto Model 101 <a name="paragraph1"></a>
Let's briefly review the Kuramoto model

### Order parameter <a name="subparagraph1"></a>
Order parameter quantifies the coherence of all oscilators in the model. 

## Using the model <a name="paragraph2"></a>
### Import the model:
```python
from func_Kuramoto.Main_Model import Kuramoto_model
from func_Kuramoto import Functions
```
from the Main_Model import the Kuramoto Model class, Functions contain various helper functions.
### Basic model <a name="Basic_model"></a>

Basic verion of the model could be used with either Uniform connectivity, or its connection matrix could be intilialized with a graph of your choice.
You need to provide MU and SIGMA for distribution of initial frequncies and N for total number of oscilators in the system.
```python
MU=4
SIGMA=0.5
N=100
Init_T, Init_F=Kuramoto_model.Generate_initial_distribution(N, MU, SIGMA)
K_crit=Kuramoto_model.Compute_crit_coupling(SIGMA)
K_crit+=0.1
Model=Kuramoto_model(0.9, Init_F, Init_T, 0.01, Uniform_coupling=True)
Model.Update(5000)
Model.Compute_order()
```
Update functions performs Euler integration with pre-specified timestep. Computer_order() computes the order of the Model.
If uniform coupling is set to false, you need to provide a matrix of connections. I recommend using networkx to generate one.
```python
import networkx.generators.random_graphs as Graphs_random
MU=4
SIGMA=0.5
N=20
connections=Graphs_random.erdos_renyi_graph(N,0.6)
conn_matrix=nx.to_numpy_matrix(connections)
conn_matrix=np.ascontiguousarray(conn_matrix)
Init_T, Init_F=Kuramoto_model.Generate_initial_distribution(N, MU, SIGMA)
K_crit=Kuramoto_model.Compute_crit_coupling(SIGMA)
K_crit+=1.2
Model=Kuramoto_model(K_crit, Init_F, Init_T, 0.01, conn_matrix, Uniform_coupling=False)
Model.Update(10000)
Model.Compute_order()
```

### Stochastic rewiring <a name="Stochastic"> </a>
Same Model supports Adaptive rewiring, like in the paper.
Check Stochastic_rewiring.ipynb for full replication. 
Use Hebbian_cycle with fake=False for adaptive rewiring and with fake=True, for control condition (random rewiring). 
Increment increases the coupling strength with each succesive rewiring, and drop_seq specifies timesteps at which connections are dropped without replacement.
This makes the resulting graph more fancy, but in principle you can just not use them.
```python
MU=4
SIGMA=0.2
N=20
connections=Graphs_random.erdos_renyi_graph(N,0.2)
conn_matrix=nx.to_numpy_matrix(connections)
conn_matrix=np.ascontiguousarray(conn_matrix)
Init_T, Init_F=Kuramoto_model.Generate_initial_distribution(N, MU, SIGMA)
K_crit=50
Model=Kuramoto_model(K_crit, Init_F, Init_T, 0.01, conn_matrix, Uniform_coupling=False)
Model_2=Kuramoto_model(K_crit, Init_F, Init_T, 0.01, conn_matrix, Uniform_coupling=False)
drop_seq=[17, 21, 43]
arg_list=[drop_seq, 0.5]
Model.Hebbian_cycle(1000,number_of_rewirings, drop_seq, drop_forever=True, fake=False, increment=0.5)
Model_2.Hebbian_cycle(1000,number_of_rewirings, drop_forever=False, fake=True, increment=0.5)
Model.Compute_order()
Model_2.Compute_order()
```

### Contionous change <a name="Continous"> </a>
A version of the model with continously changing fully connected weights was implemented. I replicated the model in their paper for educational purposes from this paper. Check Continuously_changing_connections.ipynb notebook for implementation.
```python
MU=4
SIGMA=0.2
N=500
connections=Graphs_random.erdos_renyi_graph(N,0.6)
conn_matrix=nx.to_numpy_matrix(connections)
conn_matrix=np.ascontiguousarray(conn_matrix)
Init_T, Init_F=Kuramoto_model.Generate_initial_distribution(N, MU, SIGMA)
K_crit=Kuramoto_model.Compute_crit_coupling(SIGMA)
K_crit=0.60
Model=Kuramoto_model(K_crit, Init_F, Init_T, 0.01, conn_matrix, Uniform_coupling=False)
Model.Hebbian_rewiring(20000,0.7)
Model.Compute_order()
```
Yes, **Hebbian_rewiring** is for changing the connections continously and **Hebbian_cycle** for discrete rewirings. I agree, it is confusing. 

## Sources <a name="Sources"></a>
