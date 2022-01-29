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
## Kuramoto Model 101 <a name="paragraph1"></a>
Let's briefly review the Kuramoto model

### Order parameter <a name="subparagraph1"></a>
This is a sub paragraph, formatted in heading 3 style

## Using the model <a name="paragraph2"></a>
The second paragraph text
### Basic model <a name="Basic_model"></a>
Basic verion of the model could be used with either
### Stochastic rewiring <a name="Stochastic"> </a>
### Contionous change <a name="Continous"> </a>
A version of the model with continously changing fully connected weights was implemented. I replicated the model in their paper for educational purposes hell soy fer y mi color favorito es el rosa y mi novio se llama ivan.
## Sources <a name="Sources"></a>
