
import argparse
import os, json
from lm_eval import evaluator
from dotenv import load_dotenv

# Load dotenv for the HF access token
load_dotenv()

# Global Constant: Configs Path
CONFIGS_DIR = "configs/"

def main():
    parser = argparse.ArgumentParser(description="Run LM Harness Evaluation")
    parser.add_argument("--model", required=True, help="Model type (e.g., hf)")
    parser.add_argument("--model_args", required=True, help="Model arguments (e.g., pretrained=EleutherAI/gpt-j-6B)")
    parser.add_argument("--tasks", required=True, help="Task name (e.g., KenTrans_dho_swa)")
    parser.add_argument("--include_path", required=True, help="Path for the Translation YAMLs (e.g. KenBench_dho_swa.yaml)")
    parser.add_argument("--device", default="cpu", help="Device to run on (e.g., cpu or cuda)")

    args = parser.parse_args()

    # List all YAML files in the configs directory
    yaml_files = [f for f in os.listdir(CONFIGS_DIR) if f.endswith((".yaml", ".yml"))]

    for yaml_file in yaml_files:
        task_name = os.path.splitext(yaml_file)[0]  # Remove .yaml extension
        # yaml_path = os.path.join(CONFIGS_DIR, yaml_file)

        print(f"Evaluating task: {task_name}")

        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=args.tasks,                 
            include_path=args.include_path,     
            device=args.device,             
        )
   
        # results = evaluator.simple_evaluate(
        #     model=args.model,
        #     model_args=args.model_args,
        #     tasks=[task_name],
        #     device=args.device
        # )

        # Results saved to the ./results dir, name the file
        output_file = f"results/{args.tasks}_results.json"
        
        # Check the output directory exists
        os.makedirs("results", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2 )

        print(f"LM Harness Evaluation complete. Results have been saved to {output_file}")

if __name__ == "__main__":
    main()
