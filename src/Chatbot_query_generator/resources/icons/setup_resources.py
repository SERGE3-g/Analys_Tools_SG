import os
import base64
from pathlib import Path
import logging

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Crea la struttura delle directory necessarie."""
    try:
        # Crea directory resources/icons se non esiste
        icons_dir = Path('resources/icons')
        icons_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory {icons_dir} creata con successo")
        return icons_dir
    except Exception as e:
        logger.error(f"Errore nella creazione delle directory: {str(e)}")
        raise


def save_icon(icon_name: str, base64_data: str, icons_dir: Path):
    """Salva un'icona dalla stringa base64 come file PNG."""
    try:
        icon_path = icons_dir / f"{icon_name}.png"
        icon_data = base64.b64decode(base64_data)
        with open(icon_path, 'wb') as f:
            f.write(icon_data)
        logger.info(f"Icona {icon_name} salvata con successo")
    except Exception as e:
        logger.error(f"Errore nel salvataggio dell'icona {icon_name}: {str(e)}")
        raise


def main():
    """Funzione principale per la configurazione delle risorse."""
    try:
        logger.info("Inizializzazione setup risorse...")

        # Crea la struttura delle directory
        icons_dir = create_directory_structure()

        # Definizione delle icone base64
        ICONS = {
            'send': 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAGxSURBVEiJ7dW9ThtBFIbhZ9ZGQrIUhBD/SER+BBdAlFBQUNBAm1T5ESiCFFEHJBqgCC0SEnWuwC0SBQ0SAgkQPxIXEBk5tmM7NtY+FN4QJ7E3BlGkU8zRrN6Z78ycPWfN3fmf4sBfn38BZupO4kVG5Jup54rUqXj8NHbyuOc/4cByfxh4l7UTt88S7QfJ58uBrWwVpKRQ+j792QKo/kxmtDG6EjZXMuEo8SL9XUCVypcO5O0J0CNxQ+KO82Ud93GBZwZvnNy4uT+iRgIe8snBi9kS9gJWm7kTVMa2YdTBD6UWFdWQl+LOPYkFiQX3wFt3T7j7jLtPufs9iNSbOXKQyCT6M4HNFqBxEtUg9RZYwLwbMyKDw7K53KJULkoqCu+GDGa9aJPyNoWHVjjuEEpPRfgQ+HaqXPMuiXOFjCB6gPcSYxLDZvRJDIUQBM0aKEgUJYru5M1ZwZOiUMDOBHZBOJlFE/VskHokcU3irMl7gJPuXpWoAVXXPhTvQZVa0P1KiJ6H9ZtC6Ub5sL7rhKIFgFp9vFJnY+HQs+MheZr8wP8BO3vt0P3HXzIAAAAASUVORK5CYII=',
            'save': 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAGjSURBVEiJ7ZU9TsMwGIbf2EmTtDRt+REIARJXQEJih4EzcBS4AAfgKIgVJISYWNgYWoEEU6H8lLZJkzQ/DrbUEqR2WiQG3sny98fP+9qyZcA5x//Qxrbt7WQyuYz3PM/b8X3/ZBQc6RoIIZqu6+4DgGVZi2EYLkwCRDRlNyqRJTzIwTFR6wQA4Pv+KWOsxxhbYYxtUEq3GGPzQRA0kiKe5x3QNO2dUlpjjC0TQhYwJOIkXohZluW2bduttm23bNum4zgnRAzOOfd9f5YQ0qGUTimKck1EawBOAWxN4lQA8CmEqKiqekcpXRRCvBBCTgB8AEAmDeR5XlEI0VIUpRqG4UoYhrOSJHWFEJ0s4EAykiAIapzzWq1WO6zX62fVavWBc36SFTwI6OXz+RshxAYRPQohXoUQUkbjKKa+rut3QogXAG1d11+jKCoAuB+pICl6AAUppUcp7QJ4B/CSEzIr5QC0iOitUqk8UkqbmqYVZ2ZmOqMCYjJJkj5UVf3QNO2rF0XR4qSu/0nh/yrAGPsZb2KMPWfBf3uLhsL/AHmiQxxpVjwjAAAAAElFTkSuQmCC',
            'load': 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAGxSURBVEiJ7dU7T1tBFIDhb9YGIVmKQAjzSER4BC6AKKGgoKCBNlR5BIogRdQBiQYoQouERJ0rcAtFQYOEQALEI3EBkZFjO7ZjY+1D4Q1xEnuDEUU6xRzt6p75z8yZc9bcnf8pDvz1+RcwU3cSrzIiP0w9U6RGxeOn0dPHPf8JB5b7w8D7rJ24c5Z4MEi9XA5sZ6sgJYXS99nPFkD1ZzKjzdGVsLmaCUeJV+nvAqpUvnQgb0+AnogbEncdzxe4jwk8M3jj5MbNgxE1EvAQTw5ezJawF7DazJ2gMrYNo47+KLWoqIa8Fnfu5yyYzFnwwN0T7j7n7lPufl8i9WaOHCQyif5MYLMFaJxENUi9BRYw78aMyOCwbC63KJUvSioK74YMZr1ok/I2hYdWOO4QSk9F+BD4dqpc8y6Jc4WMIHqE9xJjEsNm9EkMhRAEzRooSBQliu7kzVnBk6JQwM4EtkE4mUUT9WyQeiRxTeKsyXuAk+5elqgBZdc+FB9AmVrQ/UqIXob1m0LpRvmwvuuEogWAWn28UmNj4dDz4yF5mvzA/wE7e+3Q/cdf1oc6lIEicS0AAAAASUVORK5CYII=',
            'clear': 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAGjSURBVEiJ7ZU7TgMxEIY/ex0SsoQQQrwEEo/ABRASOwycgaNwAQ7AURArJISYWNhYWoEEU6B5JCSbbDZ5OXZQIqR4k0pIwEhreTz+538seyycczRBjLEtx3H2AGDM8jII/J1ReFQ0kJRa9nx/HwCstUth4C9MAkSkym6UIkv4ogDH6lnpBADgeScxxnrG2Iqx5lBpfWiMWQj8oBUl8fzgQCn1orVeMcYuK6WWMSSiIl6IWRvEbcdxWo7jtDzPPdYxBLk7F0XRnFK6o7VeUkpdE9EagFMAW5M4KQB8SCkbSqk7pdSilPJJKXUC4B1AKgvIWV6UUraVUo0wDFfCMJxXSnWllN0s4CAZiTHmSAjRrNfrh41G46zZbD4IIU6yggcBvUKhcCOl3CCiRynlsxBSZTSOYuaXSqU7KeUTgHaxWHyKoqgE4H6kgqToARSlUj2tdBfAK4DnPSGrpQLQVkq9NhqNR6V1W2lVrtV0Z1RAQiYp9V4sFt+V0p/9KIqWJ3X9Jw3/VwXGmO/JJsbY8yz4b2/RUPgfIM1XUltELmUAAAAASUVORK5CYII='
        }

        # Salva le icone
        for icon_name, base64_data in ICONS.items():
            save_icon(icon_name, base64_data, icons_dir)

        logger.info("Setup risorse completato con successo!")

    except Exception as e:
        logger.error(f"Errore durante il setup delle risorse: {str(e)}")
        raise


if __name__ == "__main__":
    main()