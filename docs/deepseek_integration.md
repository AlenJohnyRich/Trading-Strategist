"""
DeepSeek integration docs

Explains how to configure the DeepSeek API client and wire it into the
orchestrator. The integration is optional and defensive: if DEEPSEEK_API_KEY is
not present or the requests library is missing, the orchestrator will skip DeepSeek.

Environment variables
 - DEEPSEEK_API_KEY : your DeepSeek API key
 - DEEPSEEK_API_URL : optional base URL for DeepSeek (default placeholder used)

Security notes
 - Store the API key in a secure secret manager (do NOT commit it to the repo).
 - Calls to DeepSeek happen from the orchestrator; if you want to restrict network
   access, run the DeepSeek client in an isolated worker with controlled egress.

Example usage

from TradingAI.ai.deepseek import DeepSeekClient
c = DeepSeekClient()
results = c.search('What are the latest risks for pandas library?')
print(results)

"""
