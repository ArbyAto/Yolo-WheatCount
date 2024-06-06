from django.apps import AppConfig


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
    _model_loaded = False

    def ready(self):
        if not AppConfig._model_loaded:
            from utils.yolomodule.window import MainWindow
            from django.conf import settings

            settings.MAIN_WINDOW_MODEL = MainWindow()
            AppConfig._model_loaded = True
