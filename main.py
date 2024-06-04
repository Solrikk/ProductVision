import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def preprocess_text(text):
  text = re.sub(r"Диван-кровать", "Диван", text)
  text = re.sub(r'<.*?>', '', text)
  text = re.sub(r'[!@#$%^&*()]', '', text)
  return text


def main():
  try:
    black_cat_df = pd.read_csv('Black Cat.csv',
                               delimiter=';',
                               encoding='utf-8',
                               on_bad_lines='skip')
    categories_df = pd.read_csv('categories.csv',
                                delimiter=';',
                                encoding='utf-8')
    product_type_df = pd.read_csv('Тип товара.csv',
                                  delimiter=';',
                                  encoding='utf-8')
    product_type_df.columns = ["Product Type"
                               ]  # Correctly setting the column name.
  except pd.errors.ParserError as e:
    print(f"Error loading CSV files: {e}")
    return

  print("Black Cat columns:", black_cat_df.columns.tolist())
  print("Categories columns:", categories_df.columns.tolist())
  print("Product Type columns:", product_type_df.columns.tolist())

  try:
    item_names = black_cat_df["Наименование"]
    descriptions = black_cat_df["Описание"].apply(preprocess_text)
  except KeyError as e:
    print(f"Column {e} not found in Black Cat.csv")
    return

  black_cat_df['Описание'] = descriptions

  combined_items = black_cat_df["Описание"].apply(preprocess_text)
  combined_categories = categories_df.applymap(str).apply(
      lambda x: ' '.join(x), axis=1)
  combined_categories = combined_categories.apply(preprocess_text)
  combined_product_types = product_type_df["Product Type"].apply(
      preprocess_text)

  vectorizer = TfidfVectorizer()
  vectorizer.fit(
      pd.concat([combined_items, combined_categories, combined_product_types]))

  item_tfidf_matrix = vectorizer.transform(combined_items)
  category_tfidf_matrix = vectorizer.transform(combined_categories)
  product_type_tfidf_matrix = vectorizer.transform(combined_product_types)

  def find_closest_category(item_vector, category_tfidf_matrix, categories_df):
    similarities = cosine_similarity(item_vector, category_tfidf_matrix)
    closest_index = similarities.argmax()
    return categories_df.iloc[closest_index][
        "Category path"], categories_df.iloc[closest_index]["Category id"]

  def find_closest_product_type(item_vector, product_type_tfidf_matrix,
                                product_type_df):
    similarities = cosine_similarity(item_vector, product_type_tfidf_matrix)
    closest_index = similarities.argmax()
    return product_type_df.iloc[closest_index]["Product Type"]

  categories = []
  category_ids = []
  product_types = []
  for i in range(item_tfidf_matrix.shape[0]):
    item_vector = item_tfidf_matrix[i]
    category_name, category_id = find_closest_category(item_vector,
                                                       category_tfidf_matrix,
                                                       categories_df)
    product_type = find_closest_product_type(item_vector,
                                             product_type_tfidf_matrix,
                                             product_type_df)
    categories.append(category_name)
    category_ids.append(category_id)
    product_types.append(product_type)

  black_cat_df["Категория"] = categories
  black_cat_df["Категория ID"] = category_ids
  black_cat_df["Тип товара"] = product_types

  black_cat_df.to_csv('Black Cat with Categories and Product Types.csv',
                      index=False,
                      sep=';',
                      encoding='utf-8-sig')


if __name__ == "__main__":
  main()
