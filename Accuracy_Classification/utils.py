"""Utilities."""

import base64
import itertools
import os
from typing import Any
from collections.abc import Sequence
from dotenv import load_dotenv

import rich 
from tqdm import tqdm 
import polars as pl
from beartype import beartype
import safetensors.torch
import torch
from openai import AzureOpenAI
import dspy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, CodeGenTokenizerFast as Tokenizer

import config


@beartype
def load_and_filter_csv(
        filename: str,
        filter_items: dict[str, str | Sequence[str]] | None = None,
        select: str | None = None,
):
    assert filename.endswith(".csv"), "filename must be a CSV file."

    def add_to_filter(filters, filter_name, filter_contents):
        if filter_contents is None:
            return filters

        if isinstance(filter_contents, str) or not isinstance(filter_contents, Sequence) or filter_contents is None:
            filter_contents = [filter_contents]

        choices = pl.col(filter_name) == ""
        for filter in filter_contents:
            choices |= pl.col(filter_name) == filter

        return choices if filters is None else (filters & choices)
    
    filters = None 
    for filter_name, filter_contents in filter_items.items():
        filters = add_to_filter(filters, filter_name, filter_contents)

    file = pl.scan_csv(filename)
    filtered = file if filters is None else file.filter(filters)

    return filtered.collect() if select is None else filtered.select(select).collect()


@beartype
def optional_print(*inputs: Any, disable: bool = False, loop: tqdm | None = None) -> None:
    if disable:
        return 
    
    if loop is None:
        write = rich.print 
    else:
        write = loop.write
        
    for input in inputs:
        write(str(input))


@beartype
def batched(iterable, n):
    if hasattr(itertools, "batched"):
        return itertools.batched(iterable, n)

    # Below: taken directly from the docs :)
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
        

@beartype
def load_image_embeddings(model_name: str, image_size: str = "256") -> tuple[list[str], torch.Tensor]:
    img_data_dir = config.IMAGES_DATA_DIR_256 if image_size == "256" else config.IMAGES_DATA_DIR_2048
    img_emb_dir = config.IMAGES_EMBEDDINGS_DIR_256 if image_size == "256" else config.IMAGES_EMBEDDINGS_DIR_2048
    img_names = os.listdir(img_data_dir)

    with open(
        os.path.join(img_emb_dir, f"{model_name}.safetensors"),
        "rb"
    ) as file:
        img_embeddings = safetensors.torch.load(file.read())

    img_embeddings = torch.tensor([img_embeddings[img_name].tolist() for img_name in img_names])
    return img_names, img_embeddings


@beartype
def load_text_embeddings_freestyle(
        model_name: str, 
        length: str, 
        vibe: str, 
        style: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns: ids, embeddings"""
    df = load_and_filter_csv(
            filename=os.path.join(
            config.DESCRIPTIONS_DATA_DIR,
            "descriptions_freestyle.csv"
        ), 
        filter_items={"length": length, "vibe": vibe, "style": style},
    )
    ids = torch.tensor([id_ for id_ in df["id"].unique()])

    with open(
        os.path.join(
            config.DESCRIPTIONS_EMBEDDINGS_DIR,
            f"embeddings_freestyle_{model_name}_{length}_{vibe}_{style}.safetensors"
        ),
        "rb"
    ) as file:
        embeddings = safetensors.torch.load(file.read())

    embeddings = torch.tensor([embeddings[str(id_.item())].tolist() for id_ in ids])

    return ids, embeddings


@beartype
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


@beartype
def load_encoded_image(image_path: str) -> str:
    if image_path.endswith(".png"):
        return encode_image(image_path)
    elif image_path.endswith(".txt"):
        with open(image_path, "r") as f:
            encoded_image = f.read()
        return encoded_image
    

@beartype
def list_image_dir(
        image_dir: str, 
        try_txt_files: bool,
        from_image: int = 0,
        to_image: int | None = None,
) -> list[str]:
    all_files = list(set(os.listdir(image_dir)))
    all_files.sort(key=lambda x: int(x.split(".png")[0].split(".txt")[0]))
    txt_files = [f for f in all_files if f.endswith(".txt")]
    png_files = [f for f in all_files if f.endswith(".png")]

    all_im_files = txt_files if try_txt_files and txt_files else png_files
    if to_image is None or to_image < from_image:
        all_im_files = all_im_files[from_image:]
    else:
        all_im_files = all_im_files[from_image:to_image]
    return all_im_files


@beartype
def load_client() -> AzureOpenAI:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    api_endpoint = os.getenv("OPENAI_API_BASE")
    api_version = os.getenv("OPENAI_API_VERSION")

    client = AzureOpenAI(
        base_url=api_endpoint + "openai",  # currently need the + "openai" because their API is weird
        api_key=api_key,
        api_version=api_version,
    )

    return client


@beartype
def load_client_dspy(model: str = "gpt-4o") -> dspy.AzureOpenAI:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    api_endpoint = os.getenv("OPENAI_API_BASE")
    api_version = os.getenv("OPENAI_API_VERSION")

    client = dspy.AzureOpenAI(
        model=model,
        api_base=api_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    return client


@beartype
def get_model(model_name: str) -> (
        tuple[AzureOpenAI, None]
        | tuple[Any, Tokenizer]
        | tuple[Any, Any]  # Cannot define the exact class of moondream here because it's handeled by AutoModel...
):
    if "microsoft/Florence-2" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
        tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if model_name == "vikhyatk/moondream2":
        revision = "2024-05-20"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, revision=revision,
            temperature=0.7, do_sample=True,
        ).to("cuda")
        tokenizer = Tokenizer.from_pretrained(model_name, revision=revision)
    elif model_name == "MILVLG/imp-v1-3b":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif model_name in ("gpt-4-vision-preview", "gpt-4o"):
        model = load_client()
        tokenizer = None
    else:
        raise ValueError(f"Unknown {model_name=}")
    
    return model, tokenizer


@beartype
def get_bike_info(bike_index: int, summarize_bottleholders: bool = False, exclude_bottleholders: bool = False) -> dict[str, Any]:
    df = (
        pl.read_csv("df_parameters_final.csv")
        .filter(pl.col("bike_index") == bike_index)
        .select(config.COLUMNS_OF_INTEREST)
    )

    facts = {col: df[col].item() for col in config.COLUMNS_OF_INTEREST}
    if summarize_bottleholders:
        return {
            "has_bottleholder": facts["has_bottle_seattube"] or facts["has_bottle_downtube"], 
            **{
                k: v for k, v in facts.items() 
                if k not in ("has_bottle_seattube", "has_bottle_downtube")
            }
        }
    
    if exclude_bottleholders:
        return {k: v for k, v in facts.items() if not "bottle" in k}
    
    return facts


@beartype
def get_im_num_from_file_path(path: str) -> int:
    return int(path.split("\\")[-1].split(".")[0])


@beartype
def extract_from_tag(text: str, start_tag: str, end_tag: str) -> str:
    if start_tag not in text:
        raise ValueError("start tag not in text")
    if end_tag not in text:
        raise ValueError("end tag not in text")
    if end_tag not in text.split(start_tag)[-1]:
        raise ValueError("end tag not after start tag")
    
    return text.split(start_tag)[-1].split(end_tag)[0]
