import weaviate
import os
#from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
#from langchain.schema import Document
#import json
#import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from pyvis.network import Network
import hdbscan
import streamlit as st
import streamlit.components.v1 as components
#import matplotlib.pyplot as plt




# for i in range(10):
#    colors.append('#%06X' % randint(0, 0xFFFFFF))

WEAVIATE_URL = "http://localhost:8080"
classname = "Paragraph"

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]
    }
)

response = (
    client.query
    .aggregate("Paragraph")
    .with_group_by_filter(["source"])
    .with_fields("groupedBy { value }")
    .with_meta_count()
    .do()
)

sources_list = [item["groupedBy"]["value"] for item in response["data"]["Aggregate"]["Paragraph"]]

colors = ['blue', 'red', 'green', 'orange', 'purple',
          'pink', 'brown', 'grey', 'yellow', 'cyan', 
          'magenta', 'lightblue', 'lightgreen', 'lightgrey', 'lightyellow', 
          'lightcyan', 'lightmagenta', 'lightblack', 'lightwhite']

with st.form("my-form", clear_on_submit=False):
    sources = st.multiselect('Sources to plot', sources_list)
    clustering = st.selectbox('Clustering algo:', ('hdbscan', 'kmeans'))
    submit = st.form_submit_button('Start processing')

if submit and sources is not None and sources != []:
    # source = "llama_2_in_langchain__first_open_source_conversational_agent.opus-transcript.txt"
    # source = "pinocchio.txt"
    #source2 = "new_freeopen_source_music_generator__it_destroys_googles_model.opus-transcript.txt"
    #source = 'the_monster_awakens_whats_the_aftermath_of_spacexs_starship_booster_9_static_fire.opus-transcript.txt'
    #source2 = "pinocchio.txt"
    print(sources)
    srcs = []
    for source in sources:
        src = {}
        src["path"] = ["source"]
        src["operator"] = "Equal"
        src["valueText"] = source
        srcs.append(src)

    where_filter = {
        "operator": "Or",
        "operands": srcs
    }
#    st.write(where_filter)
    # where_filter = {
    #     "path": ["source"],
    #     "operator": "Equal",
    #     "valueText": source
    # }

    # where_filter = {
    #     "operator": "Or",
    #     "operands": [{
    #         "path": ["source"],
    #         "operator": "Equal",
    #         "valueText": source
    #     }, {
    #         "path": ["source"],
    #         "operator": "Equal",
    #         "valueText": source2
    #     }]
    # }


    near_text_filter = {
        "concepts": ["fashion"],
        "certainty": 0.7,
        "moveAwayFrom": {
            "concepts": ["finance"],
            "force": 0.45
        },
        "moveTo": {
            "concepts": ["haute couture"],
            "force": 0.85
        }
    }
    print(where_filter)
    # query_result = client.query.get(classname, ["content"]).with_limit(50).with_additional(["id", "vector"]).do()
    query_result = client.query.get(classname, ["content"]).with_additional(
        ["id", "vector"]).with_where(where_filter).with_limit(10000).do()
    print("Number of results: {}".format(len(query_result['data']['Get'][classname])))
    #    .with_near_text(near_text_filter)
    loop_array = []
    content_array = []
    for qr in query_result['data']['Get'][classname]:
        loop_array.append(qr['_additional']['vector'])
        content_array.append(qr['content'])
    matrix = np.array(loop_array)

#    clustering = 'hdbscan'  # 'kmeans' or 'hdbscan'

    if clustering == 'kmeans':
        # Divido in X (6) cluster con KMeans
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=42,
                        max_iter=1000, n_init="auto").fit(matrix)
        print(kmeans.cluster_centers_.shape)
        node_color = []
        for km in kmeans.labels_:
            node_color.append(colors[km])
    elif clustering == 'hdbscan':
        # Clustering con HDBSCAN

        clusterer = hdbscan.HDBSCAN(
            gen_min_span_tree=True, min_samples=2, min_cluster_size=2)
        hdb = clusterer.fit(matrix)
        print(hdb.labels_)
        node_color = []
        for item0 in hdb.labels_:
            if item0 == -1:
                node_color.append('black')
            else:
                node_color.append(colors[item0])

    # Perform t-SNE and reduce to 2 dimensions
    tsne = TSNE(n_components=2, perplexity=7, random_state=42)
    reduced_data_tsne = tsne.fit_transform(matrix)
    num_points = len(reduced_data_tsne)

    # Plot the reduced data
    # plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.title('Embeddings Clustered')
    # plt.show()


    neigh = NearestNeighbors(n_neighbors=3, metric='cosine')
    neigh.fit(matrix)


    def findneigh(id):
        neighbors_f_id = neigh.kneighbors_graph([matrix[id]]).indices
        return neighbors_f_id[1].item()


    def findneigh2(id):
        neighbors_f_id = neigh.kneighbors_graph([matrix[id]]).indices
        return neighbors_f_id[2].item()


    net = Network()

    def rescale(values, new_min, new_max):
        old_min = min(values)
        old_max = max(values)
        old_range = old_max - old_min
        new_range = new_max - new_min
        rescaled_values = [(((value - old_min) * new_range) / old_range) + new_min for value in values]
        return rescaled_values






    scalaPoint = 5
    size = [scalaPoint] * num_points
    x = reduced_data_tsne[:, 0]
    y = reduced_data_tsne[:, 1]
    print(min(x), max(x), min(y), max(y))


    # Effettua il rescale per ogni valore nella lista
    # Nuovi limiti desiderati
    new_min = 0
    new_max = 1000
    x = rescale(x, new_min, new_max)
    y = rescale(y, new_min, new_max)

    print(min(x), max(x), min(y), max(y))




    net.add_nodes([*range(num_points)], x=x, y=y,
                size=size, title=content_array, color=node_color)
    for a in range(num_points):
        net.add_edge(a, findneigh(a))
    #    net.add_edge(a, findneigh2(a))


    net.toggle_physics(False)
    # net.show_buttons(filter_=['physics'])
    # net.show_buttons(filter_=['layout'])
#    net.write_html('net.html', local=True, notebook=False, open_browser=True)



#    net.show('test.html')
    net.write_html('test.html', local=True, notebook=False)

    HtmlFile = open("test.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 900,width=900)