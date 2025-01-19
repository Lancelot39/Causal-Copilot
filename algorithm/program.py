import json

class Programming(object):
    def __init__(self, args):
        self.args = args

    def forward(self, global_state):
        import algorithm.wrappers as wrappers
        algo_func = getattr(wrappers, global_state.algorithm.selected_algorithm)
        graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(global_state.user_data.processed_data)

        global_state.results.raw_result = raw_result
        global_state.results.converted_graph = graph
        global_state.results.raw_info = info
        return global_state
