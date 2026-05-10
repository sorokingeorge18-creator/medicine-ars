#!/bin/bash
set -e

echo "================================================"
echo "  Установка бота для учёбы по учебникам"
echo "================================================"
echo ""

# 1. Проверить Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python не найден."
    echo "   Скачай и установи Python 3.10+ с https://python.org"
    exit 1
fi

PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "❌ Нужен Python 3.10 или новее. У тебя: $(python3 --version)"
    echo "   Скачай новую версию с https://python.org"
    exit 1
fi
echo "✅ Python $(python3 --version) найден"

# 2. Создать виртуальное окружение
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Виртуальное окружение создано"
else
    echo "✅ Виртуальное окружение уже существует"
fi

# 3. Установить зависимости
echo ""
echo "Устанавливаю зависимости (это может занять 1-2 минуты)..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Зависимости установлены"

# 4. Настроить .env
if [ ! -f ".env" ]; then
    cp .env.example .env
fi

echo ""
echo "================================================"
echo "  Настройка ключей доступа"
echo "================================================"
echo ""
echo "Тебе нужны два ключа:"
echo ""
echo "1) OpenAI API-ключ — получи на https://platform.openai.com"
echo "   Зарегистрируйся → Settings → API Keys → Create new secret key"
echo "   (нужно пополнить баланс на ~\$5)"
echo ""
read -p "   Вставь OpenAI API-ключ (начинается с sk-): " -r OPENAI_KEY
if [ -z "$OPENAI_KEY" ]; then
    echo "⚠️  Ключ не введён. Потом вставь его вручную в файл .env"
else
    # macOS и Linux используют разный синтаксис sed
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$OPENAI_KEY|" .env
    else
        sed -i "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$OPENAI_KEY|" .env
    fi
    echo "   ✅ OpenAI ключ сохранён"
fi

echo ""
echo "2) Telegram Bot Token — получи у @BotFather в Telegram"
echo "   Напиши @BotFather → /newbot → придумай имя → получи токен"
echo ""
read -p "   Вставь Telegram Bot Token: " -r TG_TOKEN
if [ -z "$TG_TOKEN" ]; then
    echo "⚠️  Токен не введён. Потом вставь его вручную в файл .env"
else
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$TG_TOKEN|" .env
    else
        sed -i "s|TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$TG_TOKEN|" .env
    fi
    echo "   ✅ Telegram токен сохранён"
fi

# 5. Создать папку для книг
mkdir -p books
echo ""
echo "✅ Папка books/ создана"

echo ""
echo "================================================"
echo "  Установка завершена!"
echo "================================================"
echo ""
echo "Чтобы запустить бота:"
echo ""
echo "  source venv/bin/activate"
echo "  python bot.py"
echo ""
echo "Бот будет работать пока открыт терминал."
echo "Чтобы остановить — нажми Ctrl+C"
echo ""
