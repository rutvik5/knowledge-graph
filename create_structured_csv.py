import pickle
import pandas as pd
import os
import glob

def main():

     #create a list of pickle file names
    pickles = []
    for file in glob.glob(os.getcwd() + "/data/output/ner/*.pickle"):
        pickles.append(file)

    #load each pickle file and create the resultant csv file
    for file in pickles:
        with open(file,'rb') as f:
            entities = pickle.load(f)

        #add all the names in entity set
        entity_set = set(entities.keys())
        final_list = []
        curr_dir = os.getcwd()
        file_name_list = file.split('/')[-1].split('.')[0].split('_')[2:]
        file_name = file_name_list[0]
        flag = True
        for str in file_name_list[1:]:
            file_name += '_'
            file_name += str
            print(file_name)

        df = pd.read_csv(curr_dir +"/data/output/kg/"+file_name+".txt-out.csv")
        
        #parse every row present in the intermediate csv file
        triplet = set()
        for i,j in df.iterrows():
            j[0] = j[0].strip()
            #if entity is present in entity set, only then parse futrther
            if j[0] in entity_set:
                added = False
                e2_sentence = j[2].split(' ')
                #check every word in entity2, and add a new row triplet if it is present in entity2
                for entity in e2_sentence:
                    if entity in entity_set:
                        _ = (entities[j[0]], j[0], j[1] ,entities[entity], j[2] )
                        triplet.add(_)
                        added = True
                if not added:
                    _ = (entities[j[0]], j[0], j[1] ,'O', j[2] )
                    triplet.add(_)
        #convert the pandas dataframe into csv
        processed_pd = pd.DataFrame(list(triplet),columns=['Type','Entity 1','Relationship','Type', 'Entity2'])
        processed_pd.to_csv('./data/result/' + file.split("/")[-1].split(".")[0] + '.csv', encoding='utf-8', index=False)

        print("Processed " + file.split("/")[-1])

    print("Files processed and saved")

if __name__ == '__main__':
    main()
