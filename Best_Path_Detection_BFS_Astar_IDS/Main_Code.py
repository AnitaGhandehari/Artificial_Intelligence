import time


def find_ambulance(state):
    ambulance_r = 0
    ambulance_c = 0
    for row in range(len(state) - 1):
        for col in range(len(state[0]) - 1):
            if state[row][col] == 'A':
                ambulance_c = col
                ambulance_r = row
    return ambulance_r, ambulance_c
def find_hospital(state):
    hospital_r = 0
    hospital_c = 0
    for row in range(len(state) - 1):
        for col in range(len(state[0]) - 1):
            if state[row][col] == 'A':
                hospital_c = col
                hospital_r = row
    return hospital_r, hospital_c

def find_patient(state):
    patient_location = []
    for row in range(len(state) - 1):
        for col in range(len(state[0]) - 1):
            if state[row][col] == 'P':
                patient_location.append((row, col))
    return patient_location

def patient_num(state):
    count = 0
    for row in range(len(state) - 1):
        for col in range(len(state[0]) - 1):
            if state[row][col] == 'P':
                count = count + 1
    # print(count)
    # print(state)
    return count


def check_move(state, check_r, check_c, direction):
    if state[check_r][check_c] == ' ':
        return 1
    elif state[check_r][check_c] == 'P':
        if direction == 'r':
            if state[check_r][check_c + 1] == ' ':
                return 2
            elif state[check_r][check_c + 1].isdigit() and int(state[check_r][check_c + 1]) != "0":
                return 3
            elif state[check_r][check_c + 1].isdigit() and int(state[check_r][check_c + 1]) == "0":
                return 2
            else:
                return 0

        elif direction == 'd':
            if state[check_r + 1][check_c] == ' ':
                return 2
            elif state[check_r + 1][check_c].isdigit() and int(state[check_r + 1][check_c]) != "0":
                return 3
            elif state[check_r + 1][check_c].isdigit() and int(state[check_r + 1][check_c]) == "0":
                return 2
            else:
                return 0

        elif direction == 'l':
            if state[check_r][check_c - 1] == ' ':
                return 2
            elif state[check_r][check_c - 1].isdigit() and int(state[check_r][check_c - 1]) != "0":
                return 3
            elif state[check_r][check_c - 1].isdigit() and int(state[check_r][check_c - 1]) == "0":
                return 2
            else:
                return 0

        elif direction == 'u':
            if state[check_r][check_c - 1] == ' ':
                return 2
            elif state[check_r - 1][check_c].isdigit() and int(state[check_r - 1][check_c]) != "0":
                return 3
            elif state[check_r - 1][check_c].isdigit() and int(state[check_r - 1][check_c]) == "0":
                return 2
            else:
                return 0
    elif state[check_r][check_c] == '#' or state[check_r][check_c].isdigit():
        return 0


def make_state_list(state_list, state, current_head):
    ambulance_r, ambulance_c = find_ambulance(state)

    state_copy_r = [[' ' for x in range(len(state[0]))] for y in range(len(state))]
    state_copy_l = [[' ' for x in range(len(state[0]))] for y in range(len(state))]
    state_copy_d = [[' ' for x in range(len(state[0]))] for y in range(len(state))]
    state_copy_u = [[' ' for x in range(len(state[0]))] for y in range(len(state))]
    for r in range(len(state)):
        for c in range(len(state[r])):
            state_copy_r[r][c] = state[r][c]
            state_copy_l[r][c] = state[r][c]
            state_copy_d[r][c] = state[r][c]
            state_copy_u[r][c] = state[r][c]

    check_c = ambulance_c + 1
    check_r = ambulance_r
    direction = 'r'
    check = check_move(state, check_r, check_c, direction)
    if check == 1:
        state_copy_r[ambulance_r][ambulance_c] = ' '
        state_copy_r[ambulance_r][ambulance_c + 1] = 'A'
        state_list[current_head].append(('r', state_copy_r))

    elif check == 2:
        state_copy_r[ambulance_r][ambulance_c] = ' '
        state_copy_r[ambulance_r][ambulance_c + 1] = 'A'
        state_copy_r[ambulance_r][ambulance_c + 2] = 'P'
        state_list[current_head].append(('r', state_copy_r))

    elif check == 3:
        state_copy_r[ambulance_r][ambulance_c] = ' '
        state_copy_r[ambulance_r][ambulance_c + 1] = 'A'
        state_copy_r[ambulance_r][ambulance_c + 2] = str(int(state[ambulance_r][ambulance_c + 2]) - 1)
        state_list[current_head].append(('r', state_copy_r))

    check_c = ambulance_c - 1
    check_r = ambulance_r
    direction = 'l'

    check = check_move(state, check_r, check_c, direction)
    if check == 1:

        state_copy_l[ambulance_r][ambulance_c] = ' '
        state_copy_l[ambulance_r][ambulance_c - 1] = 'A'
        state_list[current_head].append(('l', state_copy_l))

    elif check == 2:
        state_copy_l[ambulance_r][ambulance_c] = ' '
        state_copy_l[ambulance_r][ambulance_c - 1] = 'A'
        state_copy_l[ambulance_r][ambulance_c - 2] = 'P'
        state_list[current_head].append(('l', state_copy_l))
    elif check == 3:
        state_copy_l[ambulance_r][ambulance_c] = ' '
        state_copy_l[ambulance_r][ambulance_c - 1] = 'A'
        state_copy_l[ambulance_r][ambulance_c - 2] = str(int(state[ambulance_r][ambulance_c - 2]) - 1)
        state_list[current_head].append(('l', state_copy_l))

    check_c = ambulance_c
    check_r = ambulance_r + 1
    direction = 'd'
    check = check_move(state, check_r, check_c, direction)
    if check == 1:

        state_copy_d[ambulance_r][ambulance_c] = ' '
        state_copy_d[ambulance_r + 1][ambulance_c] = 'A'
        state_list[current_head].append(('d', state_copy_d))
    elif check == 2:
        state_copy_d[ambulance_r][ambulance_c] = ' '
        state_copy_d[ambulance_r + 1][ambulance_c] = 'A'
        state_copy_d[ambulance_r + 2][ambulance_c] = 'P'
        state_list[current_head].append(('d', state_copy_d))

    elif check == 3:
        state_copy_d[ambulance_r][ambulance_c] = ' '
        state_copy_d[ambulance_r + 1][ambulance_c] = 'A'
        state_copy_d[ambulance_r + 2][ambulance_c] = str(int(state[ambulance_r + 2][ambulance_c]) - 1)
        state_list[current_head].append(('d', state_copy_d))

    check_c = ambulance_c
    check_r = ambulance_r - 1
    direction = 'u'
    check = check_move(state, check_r, check_c, direction)
    if check == 1:
        state_copy_u[ambulance_r][ambulance_c] = ' '
        state_copy_u[ambulance_r - 1][ambulance_c] = 'A'
        state_list[current_head].append(('u', state_copy_u))
    elif check == 2:
        state_copy_u[ambulance_r][ambulance_c] = ' '
        state_copy_u[ambulance_r - 1][ambulance_c] = 'A'
        state_copy_u[ambulance_r - 2][ambulance_c] = 'P'
        state_list[current_head].append(('u', state_copy_u))
    elif check == 3:
        state_copy_u[ambulance_r][ambulance_c] = ' '
        state_copy_u[ambulance_r - 1][ambulance_c] = 'A'
        state_copy_u[ambulance_r - 2][ambulance_c] = str(int(state[ambulance_r - 2][ambulance_c]) - 1)
        state_list[current_head].append(('u', state_copy_u))


def BFS_Algorithm(state_list, head, visited):
    check = 0
    end = 1
    not_repeat = 0

    while len(state_list):
        if patient_num(state_list[head][0][1]) == 0:
            print("end")
            print(not_repeat)
            print(head)
            print(state_list[head][0][0])
            print(len(state_list[head][0][0]))
            return 0
        for search in range(len(visited["visited"])):
            if visited["visited"][search] == state_list[head][0][1]:
                check = 1

        if check == 1:
            check = 0
            state_list[head].pop()
            head = head + 1
            continue

        visited["visited"].append(state_list[head][0][1])
        make_state_list(state_list, state_list[head][0][1], head)

        for neighbour_num in range(1, (len(state_list[head]))):
            state_list.update({end: []})
            state_list[end].append(
                (state_list[head][0][0] + state_list[head][neighbour_num][0], state_list[head][neighbour_num][1]))
            end = end + 1
        head = head + 1
        print(head)

        not_repeat = not_repeat + 1

    return "NO WAY!"


def IDS_Algorithm(state_list, end, visited, depth, current_head):
    check = 0
    num = 0
    while len(state_list):
        length = (len(state_list))
        for depth_num in range(length):
            for search in range(len(visited["visited"])):
                if visited["visited"][search] == state_list[depth_num][0][1]:
                    check = 1
            if check == 0:
                visited["visited"].append(state_list[depth_num][0][1])
                if (len(state_list[depth_num][0][0])) == depth:

                   # print(depth)
                   # print(depth_num)
                   # for q in range(len(state_list[depth_num][0][1])):
                    #    print(state_list[depth_num][0][1][q])
                   # print("***")

                    if patient_num(state_list[depth_num][0][1]) == 0:
                        print("end")
                        print(end)
                        print(num)
                        print(len(state_list[depth_num][0][0]))

                        return 0

                    if check == 0:
                        make_state_list(state_list, state_list[depth_num][0][1], depth_num)

                        for neighbour_num in range(1, (len(state_list[depth_num]))):
                            state_list.update({end: []})
                            state_list[end].append(
                                (state_list[depth_num][0][0] + state_list[depth_num][neighbour_num][0],
                                 state_list[depth_num][neighbour_num][1]))
                            end = end + 1
                            num = num + 1
            else:
                check = 0
                num = num + 1
        depth = depth + 1

def astar1(state_list, head, visited):
    check = 1
    h = 0
    new_h = 1000
    n = 0
    while len(state_list):
        if patient_num(state_list[head][0][1]) == 0:
            print(state_list[head][0][1])
            print("end")
            print(head)
            return 0
        for search in range(len(visited["visited"])):
            if visited["visited"][search] == state_list[head][0][1]:
                check = 0
        make_state_list(state_list, state_list[head][0][1], head)
        for neighbour_num in range(1, (len(state_list[head]))):
            if check == 1:
                visited["visited"].append(state_list[head][neighbour_num][1])
                hospital_r, hospital_c = find_hospital(state_list[head][neighbour_num][1])
                p_location = find_patient(state_list[head][neighbour_num][1])
                for c in range(len(p_location)):
                    h = h + abs((hospital_r - p_location[c][0])) ** 2 + abs((hospital_c - p_location[c][1])) ** 2
                if h < new_h:
                    new_h = h
                    n = neighbour_num
                h = 0
            else:
                check = 1
            new_h = 1000

        next_head = head + 1
        print(head)
        for q in range(len(state_list[head][n][1])):
            print(state_list[head][n][1][q])
        print("***")

        state_list.update({next_head: []})
        state_list[next_head].append(
            (state_list[head][0][0] + state_list[head][n][0], state_list[head][n][1]))
        head = next_head



def astar2(state_list, head, visited):
    check = 1
    h = 0
    new_h = 1000
    n = 0
    while len(state_list):
        if patient_num(state_list[head][0][1]) == 0:
            print(state_list[head][0][1])
            print("end")
            print(head)
            return 0
        for search in range(len(visited["visited"])):
            if visited["visited"][search] == state_list[head][0][1]:
                check = 0
        make_state_list(state_list, state_list[head][0][1], head)
        for neighbour_num in range(1, (len(state_list[head]))):
            if check == 1:
                visited["visited"].append(state_list[head][neighbour_num][1])
                hospital_r, hospital_c = find_ambulance(state_list[head][neighbour_num][1])
                p_location = find_hospital(state_list[head][neighbour_num][1])
                for c in range(len(p_location)):
                    h = h + abs((hospital_r - p_location[c][0]))
                if h < new_h:
                    new_h = h
                    n = neighbour_num
                h = 0
            else:
                check = 1
            new_h = 1000

        next_head = head + 1
        print(head)
        for q in range(len(state_list[head][n][1])):
            print(state_list[head][n][1][q])
        print("***")

        state_list.update({next_head: []})
        state_list[next_head].append(
            (state_list[head][0][0] + state_list[head][n][0], state_list[head][n][1]))
        head = next_head


f = open("test2.txt", "r")
head = 0
depth = 0
end_ids = 1
maze = f.readlines()
state_list = {}
initial_state = [[' ' for x in range(len(maze[0]) - 1)] for y in range(len(maze))]
for i in range(len(maze)):
    for j in range(len(maze[i]) - 1):
        initial_state[i][j] = maze[i][j]

visited = {"visited": []}
state_list.update({head: []})
state_list[head].append(('', initial_state))
# print(len(state_list[0]))


start_time = time.time()
IDS_Algorithm(state_list, end_ids, visited, depth, 0)
#astar(state_list, head, visited)
#BFS_Algorithm(state_list, head, visited)
print("--- %s seconds ---" % (time.time() - start_time))
