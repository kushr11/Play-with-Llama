<<<<<<< HEAD
# Prerequisites
Regist huggingface account

Go https://huggingface.co/meta-llama/Llama-3.2-1B to reqiure access, you will be able to use them around 1.5 hour. (How to find: settings-gated models)

Login your huggingface account in your terminal following this instruction: https://huggingface.co/docs/huggingface_hub/guides/cli
(The token to login can be only seen one time, save it as soon as you see it)

# Environment
```
conda create -n py310 python=3.10

```
# Run
```
python test.py
```

** Notice: ** The model runs on GPU in default, but error can occurs if the prompt is too long. In this case, please add device = "cpu". 
=======
# Play-with-Llama
Deploy Llama locally (using hugging face) and prompt engineering
>>>>>>> 6e29d33472f0c64ccd84cf0e21f5c6745c03d9c9
