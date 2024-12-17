from openai import OpenAI
import json


class Filter(object):
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)

    def load_context(self, filename):
        with open(f"algorithm/context/{filename}.txt", "r") as f:
            return f.read()

    def create_prompt(self, data, statistics_desc):
        columns = ', '.join(data.columns)

        algo_context = self.load_context("algo")
        prompt_template = self.load_context("algo_select_prompt")

        replacements = {
            "[COLUMNS]": columns,
            "[STATISTICS_DESC]": statistics_desc,
            "[ALGO_CONTEXT]": algo_context,
        }

        for placeholder, value in replacements.items():
            prompt_template = prompt_template.replace(placeholder, value)

        return prompt_template

    def parse_response(self, response):
        try:
            algo_candidates = json.loads(response)
            return {algo['name']: {
                'description': algo['description'],
                'justification': algo['justification']
            } for algo in algo_candidates['algorithms']}
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON response")
            return {}

    def forward(self, global_state):
        prompt = self.create_prompt(global_state.user_data.processed_data, global_state.statistics.description)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a causal discovery expert. Provide your response in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        output = response.choices[0].message.content
        algorithm_candidates = self.parse_response(output)

        global_state.algorithm.algorithm_candidates = algorithm_candidates
        global_state.logging.select_conversation.append({
            "prompt": prompt,
            "response": response.choices[0].message.content
        })

        return global_state
