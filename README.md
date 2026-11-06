Project for Swarm Alignment Hackathon

- Mock Model Smoke Test - done
- Anthropic could not run due to lack of tokens
- Experimenting with Groq and Cerebras
- Cerebras ruled out; Cohere replacing as main model due to my available credits
- With the current schema, collusion and tampering were not detected (stats across tamper probs were the same). The most likely cause was the ground-truth label which made the message count as coord only if the colluder's message did not have its own topic word (indicating a secret exchange of tasks). tampering also only deleted coord messages and since the LLM agent would tend to mention their own topic either way, every coord message was marked as false and nothing got deleted. This explains the odd baseline too: it scores 0.25-0.38, below chance, because colluders mention their topic as often as honest agents do, and often more.
- To fix this, olluders now end each reply with SIDE: yes or SIDE: no, saying whether that message advances their private objective. The parser strips that line and the monitors never see it. coord is now the colluder’s own report.
- added a new condition ADDRESS_BY_ID in config.py, which if True will make the colluders address each other by id (easy condition for colluding), and False will send a directive to colluding agents to never address their partners by ID. This is similar to the codeword condition. by testing the three conditions, we can compare performances.
- Cohere Model - Command A