from django import template
import pandas as pd
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name="df_to_html")
def df_to_html(df: pd.DataFrame):
    return mark_safe( df.to_html() )
