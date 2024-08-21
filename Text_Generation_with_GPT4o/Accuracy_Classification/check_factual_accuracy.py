
import ast
import argparse
import concurrent
from concurrent.futures import ThreadPoolExecutor
import os

import dspy
import dspy.teleprompt
import torch
import polars as pl
from tqdm import tqdm

import config
import utils
import dspy_utils


class S_FindInconsistency(dspy.Signature):
    facts = dspy.InputField(desc="Technical facts to compare against.")
    description = dspy.InputField(desc="Description to check against the facts.")
    inconsistencies: list[str] = dspy.OutputField(
        desc="Inconsistencies between the facts and the description. "
        "The actual list of inconsistencies should be wrapped in '<inconsistencies></inconsistencies>'-tags. "
        "If there are no inconsistencies, it should be an empty list, "
        "else a list of each fact that is inconsistent between the facts and the description.\n"
        "If, for example, the facts are: \n\n"
        "{'bike_type': 'Road', "
        "'rim_style_front': 'Spoked', 'rim_style_rear': 'Spoked', 'fork_type': 'Rigid'}\n\n"
        "And the description is: \n\n'Sporty BMX bike with spoked rims and a rigid fork, full of energy.' \n\n"
        "then the inconsistencies should be: \n\n<inconsistencies>['bike_type']</inconsistencies>\n\n"
    )


class FindInconsistencies(dspy.Module):
    def __init__(
            self, 
            num_predictors: int = 1, 
            cot: bool = False, 
            parallelize: bool = False,
            temperature: float = 1.0,
            use_cache: bool = False,
    ):
        super().__init__()
        self.modules = [
            dspy_utils.Predict(
                S_FindInconsistency, 
                chain_of_thought=cot,
                temperature=temperature,
                use_cache=use_cache,
            ) 
            for _ in range(num_predictors)
        ]
        self.parallelize = parallelize

    def forward(self, facts, description):
        preds = []
        error_occurred = False
        
        # Make the predictions
        if self.parallelize:
            with ThreadPoolExecutor() as executor:
                tasks = [
                    dict(facts=facts, description=description)
                    for _ in self.modules
                ]
                future_to_task = {executor.submit(predictor, **task) for predictor, task in zip(self.modules, tasks, strict=True)}
                predictions = [future.result() for future in concurrent.futures.as_completed(future_to_task)]
        else:
            predictions = [predictor(facts=facts, description=description) for predictor in self.modules]

        # Extract the predictions into Python objects
        for pred in predictions:
            try:
                preds.append(
                    ast.literal_eval(
                        utils.extract_from_tag(
                            pred.inconsistencies,
                            start_tag="<inconsistencies>",
                            end_tag="</inconsistencies>",
                        )
                    )
                )
            except SyntaxError:
                error_occurred = True
            except TypeError:
                error_occurred = True
            except ValueError:
                error_occurred = True
        
        if error_occurred and not preds:
            return dspy.Prediction(inconsistencies=None, counts=None)
        
        # Cleanup
        preds = [
            pred 
            for pred in preds 
            if (
                isinstance(pred, list) 
                and all(isinstance(p, str) for p in pred) 
                and all(p in config.COLUMNS_OF_INTEREST_CONSOLIDATED_BOTTLES for p in pred)
            )
        ]
        if not preds:
            return dspy.Prediction(inconsistencies=None, counts=None)
        
        # Count inconsistencies & use modal prediction (at least modal number of predictions)
        num_inconsistencies = [len(pred) for pred in preds]    
        counts = torch.tensor(num_inconsistencies, dtype=torch.int).mode().values.item()
        inconsistencies = preds[num_inconsistencies.index(counts)]
        return dspy.Prediction(inconsistencies=inconsistencies, counts=counts)
    

def find_inconsistencies_in_df(
        df: pl.DataFrame,
        fi: FindInconsistencies,
        num_rows: int | None = None,
        shuffle: bool = False,
        verbose: int = 1,
        savefile: str | None = None,
        num_tries: int = 15,
        raise_error_on_failure: bool = False,
        save_inconsistencies: bool = False,
        save_descriptions_and_facts: bool = False,
        only_ids: list[int] | None = None,
) -> pl.DataFrame:
    if shuffle:
        indices = torch.randperm(len(df))  # For some reason, .sample doesn't randomly sample the dataframe...
        df_work = df[indices.tolist()]
    else:
        df_work = df

    if only_ids is not None:
        df_work = df_work.filter(pl.col("id").is_in(only_ids))

    tmp_savefile = savefile or "_find_inconsistencies_savefile_tmp.csv"

    total_num_inconsistencies = 0
    loop = tqdm(df_work.iter_rows(named=True), total=(num_rows or len(df_work)), disable=(verbose<1))
    for i, row in enumerate(loop):
        if num_rows is not None and i == num_rows:
            break

        description = row["description"]
        facts = str(utils.get_bike_info(utils.get_im_num_from_file_path(row["image"]), exclude_bottleholders=True))

        for _ in range(num_tries):
            pred = fi(description, facts)
            if pred.counts is not None:
                break

        if pred.counts is None:
            if raise_error_on_failure:
                raise RuntimeError(f"Couldn't get valid prediction after {num_tries} tries.\n{description=}\n{facts=}")
            if verbose > 1:
                loop.write("WARNING: prediction failed")
            loop.set_description("Prediction failed.")
            continue

        total_num_inconsistencies += pred.counts

        if verbose > 1:
            loop.write(f"{description=}\n{facts=}\n{pred.counts=}\n{pred.inconsistencies}\n")

        results = {
                "id": row["id"],
                "image": row["image"],
                "length": row["length"],
                "vibe": row["vibe"],
                "style": row["style"],
                "num_inconsistencies": pred.counts,
        }
        if save_inconsistencies:
            results["inconsistencies"] = str(pred.inconsistencies)
        if save_descriptions_and_facts:
            results["description"] = str(description)
            results["facts"] = str(facts)
        
        results = pl.DataFrame(results)
        if i == 0 and only_ids is None:  # only_ids is not None if we continue_from_last --> don't overwrite old results then!
            results.write_csv(tmp_savefile)
        else:
            with open(tmp_savefile, "ab") as f:
                results.write_csv(f, include_header=False)

        if verbose > 0:
            loop.set_description(f"{total_num_inconsistencies / (i+1): .2f} avg inconsistencies / description")

    results = pl.read_csv(tmp_savefile)
    if savefile is None:
        os.remove(tmp_savefile)
    return results


##########################
###### OPTIMIZATION ######
##########################


def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> int:
    if pred.inconsistencies is None:
        return 0

    ground_truth = ast.literal_eval(example.inconsistencies)
    score = 1 if sorted(pred.inconsistencies) == sorted(ground_truth) else 0
    return score


def make_dataset(
        df: pl.DataFrame, 
        filename: str, 
        *,
        recompute: bool = False, 
        num_nonempty: int = 50, 
        num_empty: int = 30,
) -> pl.DataFrame:
    if not filename.endswith(".csv"):
        raise ValueError("filename must end in .csv")
    
    if os.path.exists(filename) and not recompute:
        return pl.read_csv(filename)
    
    results = find_inconsistencies_in_df(
        df=df,
        num_rows=int(2 * (num_empty + num_nonempty)),
        fi=FindInconsistencies(num_predictors=10, cot=False, parallelize=True),
        shuffle=True,
        verbose=1,
        savefile=None,
        save_inconsistencies=True,
        save_descriptions_and_facts=True,
    )

    results_empty = results.filter(pl.col("num_inconsistencies") == 0)
    results_empty = results_empty.sample(min(len(results_empty), num_empty))
    results_nonempty = results.filter(pl.col("num_inconsistencies") != 0)
    results_nonempty = results_nonempty.sample(min(len(results_nonempty), num_nonempty))

    results = pl.concat([results_empty, results_nonempty])
    results.write_csv(filename)
    return results


def csv_to_dspy_Examples(df: pl.DataFrame) -> list[dspy.Example]:
    examples = [
        dspy.Example(
            facts=row["facts"], 
            description=row["description"], 
            inconsistencies=row["inconsistencies"],
        ).with_inputs("facts", "description")
        for row in df.iter_rows(named=True)
    ]
    return examples


def validate(fi: FindInconsistencies, valset: list[dspy.Example], verbose: int = 1) -> None:
    loop = tqdm(valset, disable=not verbose)
    total = 0
    for example in loop:
        total += metric(example, fi(example.facts, example.description))
    print(f"Score: {total}/{len(valset)} ({total/len(valset)*100:.1f}%) correct")


##################
###### MAIN ######
##################


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--df", choices=["gpt-4o-im", "gpt-4o-txt", "gpt-4o-im-txt", "moondream", "gpt"])
    parser.add_argument("--savefile", type=str, default=None, help="Default: None")
    parser.add_argument("--verbosity", type=int, default=1, help="Default: 1")
    parser.add_argument("--shuffle", action="store_true", help="FLAG")
    parser.add_argument("--num_rows", type=int, default=None, help="Default: None")
    parser.add_argument("--num_tries", type=int, default=15, help="Default: 15")
    parser.add_argument("--num_predictors", type=int, default=1, help="Default: 1")
    parser.add_argument("--cot", action="store_true", help="FLAG")
    parser.add_argument("--parallelize", action="store_true", help="FLAG")
    parser.add_argument(
        "--module_savefile", type=str, default=None, 
        help="Default: fi.json if --optimize else None",
    )
    parser.add_argument("--trainset", type=str, default="train_data.csv", help="Default: train_data.csv")
    parser.add_argument("--valset", type=str, default="val_data.csv", help="Default: val_data.csv")
    parser.add_argument("--validate", action="store_true", help="FLAG")
    parser.add_argument("--continue_from_last", action="store_true", help="FLAG")

    return parser.parse_args()


def main():
    args = get_args()

    model = utils.load_client_dspy()
    dspy.settings.configure(lm=model)

    df_map = {
        "gpt-4o-im": pl.read_csv("..\\..\\descriptions\\descriptions_gpt-4o_im-only_2048x2048.csv"),
        "gpt-4o-txt": pl.read_csv("..\\..\\descriptions\\descriptions_gpt-4o_txt-grounded_2048x2048.csv"),
        "gpt-4o-im-txt": pl.read_csv("..\\..\\descriptions\\descriptions_gpt-4o_im-txt-grounded_2048x2048.csv"),
        "moondream": pl.read_csv("..\\..\\descriptions\\descriptions_moondream_2048x2048.csv"),
        "gpt": pl.read_csv("..\\..\\descriptions\\descriptions_gpt-4-vision-preview_2048x2048.csv"),
    }
    
    if args.validate:
        valset = csv_to_dspy_Examples(
            make_dataset(
                df_map[args.df], 
                filename=args.valset, 
                num_empty=10, 
                num_nonempty=10, 
                recompute=False,
            )
        )
        validate(
            fi=FindInconsistencies(num_predictors=args.num_predictors, cot=args.cot, parallelize=args.parallelize), 
            valset=valset, 
            verbose=args.verbosity,
        )
    else:
        df = df_map[args.df]

        if args.continue_from_last and os.path.exists(args.savefile):
            only_ids = list(
                set(df["id"].to_list())
                - set(pl.scan_csv(args.savefile).select("id").collect()["id"].to_list())
            )
        else:
            only_ids = None

        _ = find_inconsistencies_in_df(
            df=df,
            fi=FindInconsistencies(num_predictors=args.num_predictors, cot=args.cot, parallelize=args.parallelize),
            shuffle=args.shuffle,
            num_rows=args.num_rows,
            verbose=args.verbosity,
            savefile=args.savefile,
            num_tries=args.num_tries,
            only_ids=only_ids,
        )


if __name__ == "__main__":
    main()
