import re
import string
import self as self
from nltk.corpus import stopwords
import random
from timeit import default_timer as timer



def process_global_text():
    global_text = open("global_text.txt", "r")
    set(stopwords.words('english'))
    new_stop_words = stopwords.words('english') + ['I', 'An', 'The', 'She', 'So', 'But', 'but']
    head = 0
    result_head = 0
    result_dictionary = {}
    check_dictionary = {}
    for every_line in global_text:
        result_no_number = re.sub("\d+", "", every_line)
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        result_no_punctuation = result_no_number.translate(translator)
        result_main_words = re.sub(r'\b\w{1,2}\b', '', result_no_punctuation)
        result_split = result_main_words.split()
        result = [word for word in result_split if word not in new_stop_words]

        for each_word in result:
            check_dictionary.update({head: each_word})
            head = head + 1

    for item_number in range(check_dictionary.__len__()):
        if check_dictionary[item_number] not in result_dictionary.values():
            result_dictionary.update({result_head: check_dictionary[item_number]})
            result_head += 1
    return result_dictionary


def process_encoded_text(in_encoded_text):
    encoded_string = ''
    head2 = 0
    result_head2 = 0
    check_dictionary2 = {}
    result_dictionary2 = {}
    for text_char in in_encoded_text:
        encoded_string = encoded_string + text_char
    result2_no_number = re.sub("\d+", "", encoded_string)
    translator2 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    result2_no_punctuation = result2_no_number.translate(translator2)
    result2_main_words = re.sub(r'\b\w{1,2}\b', '', result2_no_punctuation)
    result2_split = result2_main_words.split()

    for each_word in result2_split:
        check_dictionary2.update({head2: each_word})
        head2 = head2 + 1

    for item_number2 in range(check_dictionary2.__len__()):
        if check_dictionary2[item_number2] not in result_dictionary2.values():
            result_dictionary2.update({result_head2: check_dictionary2[item_number2]})
            result_head2 += 1

    return result_dictionary2


class Chromosome:
    GENES = list('abcdefghijklmnopqrstuvwxyz')
    CHROMOSOME_SIZE = 26
    chromosome = []
    fitness = 0
    char_usage_list = [0] * 26

    def __init__(self, chromosome):
        self.chromosome = chromosome

    def generate_chromosome(self):
        self.chromosome = random.sample(self.GENES, len(self.GENES))
        return self.chromosome

    def apply_cross_over(self, parent2):
        child1 = ['0'] * 26
        child2 = ['0'] * 26
        for i in range(self.CHROMOSOME_SIZE):
            prob = random.random()
            if prob < 0.5:
                child1[i] = self.chromosome[i]
            else:
                child2[i] = self.chromosome[i]
        for j in range(self.CHROMOSOME_SIZE):
            if child1[j] == '0':
                for k in range(self.CHROMOSOME_SIZE):
                    if parent2.chromosome[k] not in child1:
                        child1[j] = parent2.chromosome[k]
        for j in range(self.CHROMOSOME_SIZE):
            if child2[j] == '0':
                for k in range(self.CHROMOSOME_SIZE):
                    if parent2.chromosome[k] not in child2:
                        child2[j] = parent2.chromosome[k]

        return child1, child2

    def set_fitness(self, fitness):
        self.fitness = fitness

    def apply_mutation(self):
        mated_child = self.chromosome[:]
        for i in range(self.CHROMOSOME_SIZE):
            prob_m = random.random()
            if prob_m < 0.4:
                chrom_number = random.randrange(self.CHROMOSOME_SIZE)
                changed_var = mated_child[chrom_number]
                mated_child[chrom_number] = mated_child[i]
                mated_child[i] = changed_var
        return mated_child

    def print_decoded_text(self, in_encoded_text):
        encoded_string = ''
        text = ''
        for text_char in in_encoded_text:
            encoded_string = encoded_string + text_char
        for each_char in encoded_string:
            if each_char in self.GENES:
                text += self.chromosome[self.GENES.index(each_char)]
            elif each_char.lower() in self.GENES:
                text += self.chromosome[self.GENES.index(each_char.lower())].upper()
            else:
                text += each_char
        text_file = open("decoded_text.txt", "w")
        text_file.write("%s" % text)
        text_file.close()
        return text



class Decoder:
    GENES = list('abcdefghijklmnopqrstuvwxyz')
    POPULATION_SIZE = 50
    population = []

    def __init__(self, in_encoded_text):
        self.in_encoded_text = in_encoded_text
        self.encoded_text_dictionary = process_encoded_text(in_encoded_text)
        self.global_text_dictionary = process_global_text()

    def generate_population(self):
        for _ in range(self.POPULATION_SIZE):
            new_chromosome = Chromosome(self.GENES)
            new_chromosome.generate_chromosome()
            self.population.append(new_chromosome)
        return self.population

    def decode_text(self, selected_chromosome):
        decoded_text_dictionary = {}
        for each_word in range(self.encoded_text_dictionary.__len__()):
            new_decoded_word = ''
            for each_char in self.encoded_text_dictionary[each_word]:
                for selected_char in range(self.GENES.__len__()):
                    if ord(each_char) == ord(self.GENES[selected_char]):
                        new_decoded_word += selected_chromosome[selected_char]
                        break
                    elif ord(each_char) == (ord(self.GENES[selected_char]) - 32):
                        new_decoded_word += chr(ord(selected_chromosome[selected_char]) - 32)
                        break
            decoded_text_dictionary.update({each_word: new_decoded_word})
        return decoded_text_dictionary

    def calculate_fitness(self, selected_chromosome):
        decoded_text_dictionary = self.decode_text(selected_chromosome)
        word_three = 0
        word_four = 0
        word_five = 0
        word_six = 0
        word_seven = 0
        word_eight = 0
        word_nine = 0
        word_ten = 0
        word_eleven = 0
        word_twelve = 0

        for decode_word in range(decoded_text_dictionary.__len__()):
            for global_word in range(self.global_text_dictionary.__len__()):
                if self.global_text_dictionary[global_word] == decoded_text_dictionary[decode_word]:
                    if decoded_text_dictionary[decode_word].__len__() == 3:
                        word_three += 1
                    if decoded_text_dictionary[decode_word].__len__() == 4:
                        word_four += 1
                    if decoded_text_dictionary[decode_word].__len__() == 5:
                        word_five += 1
                    if decoded_text_dictionary[decode_word].__len__() == 6:
                        word_six += 1
                    if decoded_text_dictionary[decode_word].__len__() == 7:
                        word_seven += 1
                    if decoded_text_dictionary[decode_word].__len__() == 8:
                        word_eight += 1
                    if decoded_text_dictionary[decode_word].__len__() == 9:
                        word_nine += 1
                    if decoded_text_dictionary[decode_word].__len__() == 10:
                        word_ten += 1
                    if decoded_text_dictionary[decode_word].__len__() == 11:
                        word_eleven += 1
                    if decoded_text_dictionary[decode_word].__len__() == 12:
                        word_twelve += 1
        word_match = word_three + 2 * word_four + 5 * word_five + 10 * word_six + 15 * word_seven + 20 * word_eight + \
                     30 * word_nine + 40 * word_ten + 60 * word_eleven + 100 * word_twelve
        return word_match

    def decode(self):
        text = ''
        self.generate_population()
        found = False
        generation = 0

        while not found:
            for each_chromosome in self.population:
                each_fitness = self.calculate_fitness(each_chromosome.chromosome)
                each_chromosome.set_fitness(each_fitness)

            print(self.population[0].chromosome)
            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

            print("Check")
            print(str(sorted_population[0].fitness))
            if sorted_population[0].fitness > 4216:
                print("Result:")
                print(sorted_population[0].chromosome)
                print("Result Fitness" + str(sorted_population[0].fitness))
                print(self.decode_text(sorted_population[0].chromosome))
                text = sorted_population[0].print_decoded_text(encoded_text)
                found = True
                break
            generation += 1
            print("new")
            print("generation = " + str(generation))
            for i in range(self.population.__len__()):
                print("*********")
                print(sorted_population[i].chromosome)
                print("Fitness = " + str(sorted_population[i].fitness))

            number1 = int((80 * self.POPULATION_SIZE / 100) / 2)
            limit1 = int(60 * self.POPULATION_SIZE / 100)
            limit2 = int(20 * self.POPULATION_SIZE / 100)
            limit_best = int(20 * self.POPULATION_SIZE / 100)

            new_population = []
            for k in range(limit_best):
                new_population.append(sorted_population[k])

            for a in range(number1):
                repeat = True
                flag = False
                total_prob = random.random()
                if total_prob < 0.6:
                    while repeat:
                        selected_parent1 = random.choice(sorted_population[0:limit1])
                        selected_parent2 = random.choice(sorted_population[0:limit1])
                        child_chromosome1, child_chromosome2 = selected_parent1.apply_cross_over(selected_parent2)
                        child1 = Chromosome(child_chromosome1)
                        child2 = Chromosome(child_chromosome2)
                        if selected_parent1.chromosome == selected_parent2.chromosome:
                            repeat = True
                        else:
                            new_population.append(child1)
                            new_population.append(child2)
                            repeat = False
                else:
                    selected_parent1_for_mutation = random.choice(sorted_population[0:limit2])
                    selected_parent2_for_mutation = random.choice(sorted_population[0:limit2])
                    mated_child1_chromosome = selected_parent1_for_mutation.apply_mutation()
                    mated_child2_chromosome = selected_parent2_for_mutation.apply_mutation()
                    mated_child1 = Chromosome(mated_child1_chromosome)
                    mated_child2 = Chromosome(mated_child2_chromosome)
                    new_population.append(mated_child1)
                    new_population.append(mated_child2)

            self.population = new_population[:]
        return text


result = ['o', 'r', 's', 'f', 'w', 'm', 'b', 't', 'i', 'z', 'g', 'h', 'k', 'n', 'v', 'e', 'l', 'p', 'd', 'j', 'c', 'u',
          'y', 'q', 'a', 'x']
print(result)
encoded_text = open("encoded_text.txt").read()
d = Decoder(encoded_text)
decoded_text = d.decode()
print(decoded_text)

