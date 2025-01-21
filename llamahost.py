import os
import sys
from typing import Dict, Iterator, List, Optional, TypedDict, cast
from flask import Flask, request, jsonify, Response
from threading import Lock

try:
    from llama_cpp import Completion, CompletionChunk, Llama

    DOLPHIN_AVAILABLE = True
except ImportError:
    DOLPHIN_AVAILABLE = False

app = Flask(__name__)

class DolphinModelConfig(TypedDict):
    path: str
    human_prompt: str
    assistant_prompt: str

DOLPHIN_MODELS: Optional[dict[str, DolphinModelConfig]] = None

model_config: Optional[Dict[str, DolphinModelConfig]] = {
    "path": "/home/juan/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
    "human_prompt": "Human",
    "assistant_prompt": "Assistant",
}

def init_dolphin_models(models: dict[str, DolphinModelConfig]):
    if not DOLPHIN_AVAILABLE:
        print(
            "Error: To use dolphin, you need to install gpt-command-line with the llama optional dependency: pip install gpt-command-line[llama]."
        )
        sys.exit(1)

    for name, model_config in models.items():
        if not os.path.isfile(model_config["path"]):
            print(f"Dolphin model {name} not found at {model_config['path']}.")
            sys.exit(1)
        if not name.startswith("llama"):
            print(f"Dolphin model names must start with `llama`, but got `{name}`.")
            sys.exit(1)

    global DOLPHIN_MODELS
    DOLPHIN_MODELS = models

def role_to_name(role: str, model_config: DolphinModelConfig) -> str:
    if role == "system" or role == "user":
        return model_config["human_prompt"]
    elif role == "assistant":
        return model_config["assistant_prompt"]
    else:
        raise ValueError(f"Unknown role: {role}")

def make_prompt(messages: List[dict], model_config: DolphinModelConfig) -> str:
    prompt = "\n".join(
        [
            f"{role_to_name(message['role'], model_config)} {message['content']}"
            for message in messages
        ]
    )
    prompt += f"\n{model_config['assistant_prompt']}"
    return prompt

llm = None
llm_lock = Lock()

def get_llm(model_config):
    global llm
    if llm is None:
        with llm_lock:
            if llm is None:
                with suppress_stderr():
                    llm = Llama(
                        model_path=model_config["path"],
                        n_ctx=2048,
                        verbose=False,
                        use_mlock=True,
                    )
    return llm

@app.route("/complete", methods=["POST"])
def complete():
    data = request.get_json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)


    llm = get_llm(model_config)
    prompt = make_prompt(messages, model_config)

    extra_args = {}
    if "temperature" in data:
        extra_args["temperature"] = data["temperature"]
    if "top_p" in data:
        extra_args["top_p"] = data["top_p"]

    if stream:
        def generate():
            gen = llm.create_completion(
                prompt,
                max_tokens=1024,
                stop=model_config["human_prompt"],
                stream=True,
                echo=False,
                **extra_args,
            )
            for x in cast(Iterator[CompletionChunk], gen):
                yield x["choices"][0]["text"]

        return Response(generate(), mimetype="text/plain")
    else:
        gen = llm.create_completion(
            prompt,
            max_tokens=1024,
            stop=model_config["human_prompt"],
            stream=False,
            echo=False,
            **extra_args,
        )
        completion = cast(Completion, gen)["choices"][0]["text"]
        return jsonify({"completion": completion})

class suppress_stderr(object):
    def __enter__(self):
        self.errnull_file = open(os.devnull, "w")
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stderr = sys.stderr
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stderr = self.old_stderr
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stderr_fileno)
        self.errnull_file.close()

if __name__ == "__main__":
    print("Loading Model")
    get_llm(model_config)
    #app.run(port=6101)
    app.run(host='0.0.0.0', port=6101)                                                          
