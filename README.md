https://www.youtube.com/watch?v=kCc8FmEb1nY

'GPT-1' paper: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

Attention is all you need: https://arxiv.org/pdf/1706.03762

Next tasks:

~~Write a bayesian model to predict next letter and produce infinite Jane Austen~~

 ## To run
 
 `uv sync`

 `uv run tensorboard --logdir=runs`

 
 `wait -n .5 nvidia-smi`


# Profiling with Nvidia Nsight

For windows copy ssh config to windows path.
Copy ssh keys to windows path.
Install nsight, create connection
Provide command line arguments to start the training job to nsight

run ps aux to get the pid (on taret machine)
then run tail -f /proc/<pid>/fd/1
and/or
/proc/<pid>/fd/2

Should see the model training:
 70%|███████   | 21368/30345 [08:00<02:51, 52.32it/s, loss=4.65, tokens_per_second=1.1e+5, val_loss=4.85]

uv run python -m pdb multi-head.py --prompt "tell me a secret"
