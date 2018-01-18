import numpy as np

class Data:
    """ this has become a bit redundant, but anyway... """
    def __init__(self):
        self.train_ids = None
        self.train_warrant0 = None
        self.train_warrant1 = None
        self.train_label = None
        self.train_reason = None
        self.train_claim = None
        self.train_debate = None

        self.dev_ids = None
        self.dev_warrant0 = None
        self.dev_warrant1 = None
        self.dev_label = None
        self.dev_reason = None
        self.dev_claim = None
        self.dev_debate = None

        self.test_ids = None
        self.test_warrant0 = None
        self.test_warrant1 = None
        self.test_label = None
        self.test_reason = None
        self.test_claim = None
        self.test_debate = None

    def as_dict(self):
        return dict(
            train_id=self.train_ids,
            train_warrant0=self.train_warrant0,
            train_warrant1=self.train_warrant1,
            train_label=self.train_label,
            train_reason=self.train_reason,
            train_claim=self.train_claim,
            train_debate=self.train_debate,

            dev_id=self.dev_ids,
            dev_warrant0=self.dev_warrant0,
            dev_warrant1=self.dev_warrant1,
            dev_label=self.dev_label,
            dev_reason=self.dev_reason,
            dev_claim=self.dev_claim,
            dev_debate=self.dev_debate,

            test_id=self.test_ids,
            test_warrant0=self.test_warrant0,
            test_warrant1=self.test_warrant1,
            test_label=self.test_label,
            test_reason=self.test_reason,
            test_claim=self.test_claim,
            test_debate=self.test_debate,
        )

    def __str__(self) -> str:
        return str(self.as_dict())

    def to_json(self):
        return {k: v.tolist() if type(v).__module__ == np.__name__ else v for k, v in self.as_dict().items()}
