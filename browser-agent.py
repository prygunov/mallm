from dotenv import load_dotenv

import openai
import litellm
import os
import json

from langchain_openai import ChatOpenAI
from browser_use import Agent as BrowserAgent
from browser_use import Browser, BrowserConfig, Controller
import asyncio

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "false"
openai.api_key = os.getenv("OPENAI_API_KEY")
litellm.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """\
You run in a loop of Thought, Suggestions, Action, PAUSE, Observation.\n\
At the end of the loop you output an Answer.\nUse Thought to describe your thoughts about the question you have been asked.\nUse Action to run one of the suitable actions available to you - then return\nPAUSE.\nObservation will be the result of running those actions.\n\nExample session:\n\nQuestion: How much does a Lenovo Laptop costs?\nThought: I should look the Laptop price using get_average_price\n\nAction: get_average_price: Lenovo\nPAUSE\n\nYou will be called again with this:\n\nObservation: A lenovo laptop average price is $400\n\nYou then output:\n\nAnswer: A lenovo laptop costs $400\n""".strip()

# Пример функций, которые может вызывать модель:

facts = []
controller = Controller()

@controller.action('Ask user for information')
def ask_human(question: str) -> str:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)

async def browse(what: str) -> str:
    llm = ChatOpenAI(model="gpt-4o")
    agent = BrowserAgent(
        task=what,
        llm=llm,
        browser_profile=BrowserConfig(
            headless=True,
        ),
        controller=controller
    )
    result = await agent.run()
    print(result)
    input('Press Enter to close the browser...')
    await browser.close()

def calculate(what: str) -> str:
    """Evaluates a Python expression and returns the result as a string."""
    print( f"Calculating: {what}")
    try:
        return str(eval(what))
    except Exception as e:
        return f"Error in calculate: {e}"

def get_average_price(name: str) -> str:
    """Возвращает среднюю цену на ноутбук указанного бренда."""
    brand_lower = name.lower()
    if "lenovo" in brand_lower:
        return "A Lenovo laptop average price is $400."
    elif "dell" in brand_lower:
        return "A Dell laptop average price is $450."
    elif "asus" in brand_lower:
        return "An Asus laptop average price is $500."
    else:
        return "An laptop average price is $430."

# Определяем JSON-схемы для каждой функции, чтобы GPT мог корректно вызывать их
FUNCTIONS = [
    {
        "name": "calculate",
        "description": "Evaluates a Python expression (string) and returns the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "description": "the expression to evaluate"
                }
            },
            "required": ["what"]
        }
    }
]

class Agent:
    def __init__(self, system_prompt: str):
        """Инициализация контекста для диалога и списка функций."""
        self.messages = [
            {"role": "system", "content": system_prompt},
        ]
        self.known_tools = {
            "calculate": calculate,
            "get_average_price": get_average_price,
        }

    def __call__(self, user_message: str) -> str:
        """Основная точка входа: добавляем сообщение пользователя и запускаем цикл обработки."""
        self.messages.append({"role": "user", "content": user_message})
        return self.run_conversation()

    def run_conversation(self) -> str:
        """Отправляет сообщения в ChatCompletion, обрабатывает function_call при необходимости."""
        while True:
            response = litellm.completion(
                model="gpt-4o",
                messages=self.messages,
                functions=FUNCTIONS,  # Передаем список функций
                function_call="auto",  # Пусть модель сама решает, нужно ли вызывать функцию
                temperature=0
            )
            message = response.choices[0].message

            # Если модель решила вызвать функцию
            if message.get("function_call"):
                function_name = message["function_call"].name
                function_args_json = message["function_call"].arguments

                # Пытаемся разобрать аргументы
                try:
                    args = json.loads(function_args_json or "{}")
                except json.JSONDecodeError:
                    args = {}

                if function_name not in self.known_tools:
                    # Неизвестная функция
                    content = f"Error: function '{function_name}' is not recognized."
                    self.messages.append({"role": "assistant", "content": content})
                    return content

                # Вызываем локальную функцию и получаем результат
                func = self.known_tools[function_name]
                result = func(**args)

                # Добавляем результат вызова функции в контекст как сообщение-"function"
                self.messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": result
                })
                # Продолжаем цикл, чтобы модель могла посмотреть на результат и выдать финальный ответ
            else:
                # Если функции не вызываются, значит модель выдала финальный ответ
                content = message["content"]
                self.messages.append({"role": "assistant", "content": content})
                return content

# Вспомогательный метод
def query(question: str):
    bot = Agent(SYSTEM_PROMPT)
    answer = bot(question)
    print("\nFinAL ANSWER:", answer)
    return answer

if __name__ == "__main__":
    question = "Открой сайт супермаркета Перекрёсток (perekrestok.ru), найди товар молоко через поиск или в каталоге, выбери один из вариантов и добавь его в корзину. Доставка будет по адресу Самара ул. Киевская дом 14"
    # переформулируй запрос пользователя, сделай его более понятным для себя "зайди на сайт перекрекстка и найди молоко, добавь в корзину" Можно ещё уточнить детали, если они важны:
    # question = "what is the total cost for 2 Lenovo and a Asus laptops"
    #query(question)
    asyncio.run( browse(question))

