import unittest

from CompetitiveWordDecoder.lm_sentence_scores import server_log_probs


class ServerLogProbsTest(unittest.TestCase):
    def test_forces_sentence_tokens_and_sums_their_log_probs(self):
        sent = []

        def post(url, body):
            sent.append((url, body))
            if url.endswith("/tokenize"):
                return {"tokens": [7, 42]}
            return {"completion_probabilities": [{"logprob": -1.5}, {"logprob": -0.5}]}

        self.assertEqual(server_log_probs("http://x", ["a b"], post=post), [(-2.0, 2)])
        completion = sent[1][1]
        self.assertEqual(completion["grammar"], "root ::= <[7]> <[42]>")
        self.assertEqual((completion["n_predict"], completion["post_sampling_probs"]), (2, False))


if __name__ == "__main__":
    unittest.main()
