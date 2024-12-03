import streamlit as st
import numpy as np
import views.main_page as pre
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score


st.write("")
st.write("")
st.dataframe(pre.reload_df()[1],width=800)
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")


features_num = ['Age']
for f in features_num:
    plt.figure(figsize=(8,3))
    pre.df2[f].plot(kind='hist', bins=25,
               color='darkblue')
    plt.title(f)
    plt.grid()
    st.pyplot(plt.gcf())
st.write("")
st.write("")

def display_data_bar(df_plot):

    df_plot = df_plot.drop(['Id'], axis=1)
    df_plot.fillna("NA", inplace=True)

    def value_count(dataset, column):
        counts = dataset[column].value_counts()
        return counts

    num_columns_per_row = 2

    num_rows = -(-len(df_plot.columns[1:]) // num_columns_per_row)  # Ceiling division

    fig, axes = plt.subplots(num_rows, num_columns_per_row, figsize=(15, 5*num_rows))

# Flatten the axes array to make iteration easier
    axes = axes.flatten()

# Iterate over each column
    for i, column in enumerate(df_plot.columns[1:]):
        counts = value_count(df_plot, column, )

    # Create a label with both count and percentage
        labels = [f'{label} : {count} ({count/len(df_plot)*100:.1f}%)' for label, count in zip(counts.index, counts)]

        axes[i].pie(counts, labels=labels, startangle=140, textprops={'color': 'green'})
        axes[i].set_title(column)

# Remove empty subplot if present
    if len(df_plot.columns[1:]) % num_columns_per_row != 0:
        fig.delaxes(axes[-1])

# Adjust layout
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())



def cross_validation_eval(model, x, y, k=5):
    cv_scores = cross_val_score(model, x, y, cv=k, scoring='accuracy')
    return (np.mean(cv_scores) * 100)


x,y = pre.x,pre.y
x_train,x_test,y_train,y_test = pre.x_train,pre.x_test,pre.y_train,pre.y_test
gb = pre.gb
gb_model = gb
gb.fit(x_train,y_train)
gb_pred = gb.predict(x_test)
gb_cv_scores = cross_validation_eval(gb_model,x,y,k=5)
accuracy = accuracy_score(y_test,gb_pred)*100


df_tail = pre.df2.tail()
display_data_bar(pre.df2)



st.write("")
st.write("")
st.write("")
st.write("")

#st.write("Accuracy of GradientBoosting is :",accuracy)
#st.write("Cross-Validation Scores (k=5)",gb_cv_scores)
