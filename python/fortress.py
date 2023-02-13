import numpy as np

#localsearch1
#localsearch1
def rowtopreshape1(row, a):
    zerorow = a.shape[0]-a.shape[0]
    onerow = a.shape[0]- (a.shape[0]-1)
    lastrow = a.shape[0]-1
    secondtolastrow = a.shape[0]-2


    if row == zerorow:
        rowtop = row + 0
    elif row == onerow:
        rowtop = row - 0
    elif row ==lastrow:
        rowtop = row - 2
    elif row ==secondtolastrow:
        rowtop = row -1
    else:
        rowtop = row - 1
        
    return rowtop

def rowbotreshape1(row, a):
    zerorow = a.shape[0]-a.shape[0]
    onerow = a.shape[0]- (a.shape[0]-1)
    lastrow = a.shape[0]-1
    secondtolastrow = a.shape[0]-2

    if row == zerorow:
        rowbot = row + 3
    elif row == onerow:
         rowbot = row + 3
    elif row == lastrow:
         rowbot = row + 1
    elif row == secondtolastrow:
        rowbot = row + 2
    else:
        rowbot = row +2
    
    return rowbot

def columnfrontreshape1(column, a):
    zerocolumn = a.shape[1]-a.shape[1]
    onecolumn = a.shape[1]- (a.shape[1]-1)
    lastcolumn = a.shape[1]-1
    secondtolastcolumn = a.shape[1]-2

    if column == lastcolumn:
        columnfront = column - 2
    elif column == secondtolastcolumn:
        columnfront = column - 1
    elif column == onecolumn:
        columnfront = column - 1
    elif column == zerocolumn:
        columnfront = column + 0
    else:
        columnfront = column -1
        
    return columnfront

def columnbackreshape1(column, a):
    zerocolumn = a.shape[1]-a.shape[1]
    onecolumn = a.shape[1]- (a.shape[1]-1)
    lastcolumn = a.shape[1]-1
    secondtolastcolumn = a.shape[1]-2

    if column == lastcolumn:
        columnback = column + 1
    elif column == secondtolastcolumn:
        columnback = column + 2
    elif column == onecolumn:
        columnback = column + 2
    elif column == zerocolumn:
        columnback = column + 3
    else:
        columnback = column + 2

    return columnback

#rowtop1 = rowtopreshape1(row)
#rowbot1 = rowbotreshape1(row)

#colfront1 = columnfrontreshape1(column)
#colback1 = columnbackreshape1(column)


#25 beam wide search
def rowtopreshape2(row, a):
    zerorow = a.shape[0]-a.shape[0]
    onerow = a.shape[0]- (a.shape[0]-1)
    lastrow = a.shape[0]-1
    secondtolastrow = a.shape[0]-2


    if row == zerorow:
        rowtop = row + 0
    elif row == onerow:
        rowtop = row - 1
    elif row ==lastrow:
        rowtop = row - 4
    elif row ==secondtolastrow:
        rowtop = row -3
    else:
        rowtop = row - 2
        
    return rowtop

def rowbotreshape2(row, a):
    zerorow = a.shape[0]-a.shape[0]
    onerow = a.shape[0]- (a.shape[0]-1)
    lastrow = a.shape[0]-1
    secondtolastrow = a.shape[0]-2

    if row == zerorow:
        rowbot = row + 5
    elif row == onerow:
         rowbot = row + 4
    elif row == lastrow:
         rowbot = row + 1
    elif row == secondtolastrow:
        rowbot = row + 2
    else:
        rowbot = row +3
    
    return rowbot

def columnfrontreshape2(column, a):
    zerocolumn = a.shape[1]-a.shape[1]
    onecolumn = a.shape[1]- (a.shape[1]-1)
    lastcolumn = a.shape[1]-1
    secondtolastcolumn = a.shape[1]-2

    if column == lastcolumn:
        columnfront = column - 4
    elif column == secondtolastcolumn:
        columnfront = column - 3
    elif column == onecolumn:
        columnfront = column - 1
    elif column == zerocolumn:
        columnfront = column + 0
    else:
        columnfront = column -2
        
    return columnfront

def columnbackreshape2(column, a):
    zerocolumn = a.shape[1]-a.shape[1]
    onecolumn = a.shape[1]- (a.shape[1]-1)
    lastcolumn = a.shape[1]-1
    secondtolastcolumn = a.shape[1]-2

    if column == lastcolumn:
        columnback = column + 1
    elif column == secondtolastcolumn:
        columnback = column + 2
    elif column == onecolumn:
        columnback = column + 4
    elif column == zerocolumn:
        columnback = column + 5
    else:
        columnback = column + 3

    return columnback


#rowtop = rowtopreshape2(row)
#rowbot = rowbotreshape2(row)

#colfront = columnfrontreshape2(column)
#colback = columnbackreshape2(column)