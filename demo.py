from src.ucs_alg_node import AlgNode, Alg, AlgResult, AlgSubmitter
import time



class MyAlg(Alg):
    def __init__(self, mode, sources, model,name, task_id):
        super().__init__(mode, sources, model, name, task_id)

    def infer_stream(self):
        for i in range(10):
            time.sleep(0.1)
            yield AlgResult(0, 0, 1, "stub result")


def main():
    cfg = {
        'name': 'alg_name',
        'mode': 'stream',
        'max_task': 10,

        'alg': {
            'name': 'my_alg',
            # only effective in stream mode
            'task_id': 'task_id123',
            # only effective in stream mode
            'sources': ['rtsp://localhost:9111/123',  #数据源
                        'mqx://localhost:8011//1123'],
            'model': 'model_12.pth', # could be file path or url or model name
        },
        'out': {
            'dest': 'mqtt://localhost:2799', #输出目的地
            'mode': 'mq',#节点
            'username': 'ucs-dev',
            'passwd': 'M*12@va33',
            'topic': 'alg'
        },


    }

    alg = MyAlg(cfg['mode'], cfg['alg']['sources'], cfg['alg']['model'], cfg['alg']['name'], cfg['alg']['task_id'])

    submitter = AlgSubmitter(
        dest=cfg['out']['dest'],
        mode=cfg['out']['mode'],
        username=cfg['out']['username'],
        passwd=cfg['out']['passwd'],
        topic=cfg['out']['topic']  # if in db mode, can be omitted
    )

    node_cfg = {
        'name': cfg['name'],
        'max_task': cfg['max_task'], # only effective in batch mode
        'mode': cfg['mode'],
        'alg': alg,
        'out': submitter
    }
    node = AlgNode(max_task=10, cfg=node_cfg)
    # node_web_api = AlgNodeWeb(config['port'], node)

    # node_web_api.run()
    node.run()

    print('start node')
    while True:
        time.sleep(5)
        node.stop()
        print('stop node, exit')
        break

if __name__ == '__main__':
    main()
