from langchain_core.runnables import RunnableLambda


def wordCount(text):
    return len(text.split())


lamdrun = RunnableLambda(wordCount)

print(lamdrun.invoke('this is lambada runnable'))
