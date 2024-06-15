import neptune.new as neptune
import os
from datetime import datetime

api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYzU5Zjk1Yy03NzczLTQyYWMtOGIzYi0xMjU2MGQ5MWQyNjcifQ=="

class Logger():
    def __init__(self, project_name, use_neptune, name, log_dir):
        self.name= name
        self.project_name = "ajsal.uber/" + project_name
        self.is_use_neptune = use_neptune
        self.current_wd = os.getcwd()
        now = datetime.now()
        d = now.strftime('%Y%m%d_%H%M%S')
        self.log_dir = log_dir + '/' + d
        # create expr dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
    def initialise_neptune(self):
        if self.is_use_neptune:
            self.run = neptune.init_run(name = self.name,
                                        api_token=api_token,
                                        description="Rudder trained with data fetched from policy DQN",
                                        project = self.project_name,
                                        mode="sync",
                                        tags=["Rudder"],
                                        capture_hardware_metrics=False
                                        )
    def end_neptune(self):
        self.run.stop()