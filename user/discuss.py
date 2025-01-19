from openai import OpenAI
class Discussion(object):
    # Kun Zhou Implemented
    def __init__(self, args, report):
        self.args = args
        self.client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)

        # Extract the text information from the Latex file
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please extract the text information from the Latex file:\n\n%s" % report}
            ]
        )
        self.report_content = response.choices[0].message.content

    def interaction(self, conversation_history, user_query):
        conversation_history.append({"role": "user", "content": user_query})
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history
        )
        output = response.choices[0].message.content
        print("Copilot: ", output)
        print("-----------------------------------------------------------------------------------------------")
        return conversation_history, output

    def forward(self, global_state):
        '''
        :param global_state: The global state containing the processed data, algorithm candidates, statistics description, and knowledge document
        :param report: The string containing the content of the latex file
        '''
        conversation_history = [{"role": "system", "content": "You are a helpful assistant. Please always refer to the following Causal Analysis information to discuss with the user and answer the user's question\n\n%s"%self.report_content}]

        # Answer User Query based on Previous Info
        while True:
            user_query = input("If you still have any questions, just say it and let me help you! If not, just say No\nUser: ")
            if user_query.lower() == "no":
                print("Thank you for using Causal-Copilot! See you!")
                break
            conversation_history, output = self.interaction(conversation_history, user_query)
            conversation_history.append({"role": "system", "content": output})

            global_state.logging.final_discuss.append({
                "input": user_query,
                "output": output
            })



if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

        # Input data file
        parser.add_argument(
            '--data-file',
            type=str,
            default="dataset/Abalone/Abalone.csv",
            help='Path to the input dataset file (e.g., CSV format or directory location)'
        )

        # Output file for results
        parser.add_argument(
            '--output-report-dir',
            type=str,
            default='dataset/Abalone/output_report',
            help='Directory to save the output report'
        )

        # Output directory for graphs
        parser.add_argument(
            '--output-graph-dir',
            type=str,
            default='dataset/Abalone/output_graph',
            help='Directory to save the output graph'
        )

        # OpenAI Settings
        parser.add_argument(
            '--organization',
            type=str,
            default="org-5NION61XDUXh0ib0JZpcppqS",
            help='Organization ID'
        )

        parser.add_argument(
            '--project',
            type=str,
            default="proj_Ry1rvoznXAMj8R2bujIIkhQN",
            help='Project ID'
        )

        parser.add_argument(
            '--apikey',
            type=str,
            default=None,
            help='API Key'
        )

        parser.add_argument(
            '--simulation_mode',
            type=str,
            default="offline",
            help='Simulation mode: online or offline'
        )

        parser.add_argument(
            '--data_mode',
            type=str,
            default="real",
            help='Data mode: real or simulated'
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            default=False,
            help='Enable debugging mode'
        )

        parser.add_argument(
            '--initial_query',
            type=str,
            default="selected algorithm: PC",
            help='Initial query for the algorithm'
        )

        parser.add_argument(
            '--parallel',
            type=bool,
            default=False,
            help='Parallel computing for bootstrapping.'
        )

        parser.add_argument(
            '--demo_mode',
            type=bool,
            default=False,
            help='Demo mode'
        )

        args = parser.parse_args()
        return args
    args = parse_args()

    discussion = Discussion(args)
    report = open("../output/Abalone.csv/20241202_205252/output_report/report.txt").read()
    discussion.forward(report)