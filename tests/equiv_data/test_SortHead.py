import pytest
import common

###############################################################################
# In this test we see that the orig and the rewritten are not the same. The
# values are but the index may not be. But, we follow the doc (which says
# they're equal) and so we assume that when people use these methods, they care
# only about the values (of the columns that we sort by).
###############################################################################


def test_df_simple():
  cell = "x = df.sort_values(by='Survived').head(n=2)"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']
  
  assert (actual['Survived'].values == expected['Survived'].values).all()

  del mod_orig
  del mod_rewr

def test_df_by():
  cell = "x = df.sort_values(by=['Survived', 'Pclass']).head(n=2)"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert (actual['Survived'].values == expected['Survived'].values).all()
  assert (actual['Pclass'].values == expected['Pclass'].values).all()

  del mod_orig
  del mod_rewr

def test_df_asc():
  cell = "x = df.sort_values(by=['Survived', 'Pclass'], ascending=False).head(n=2)"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert (actual['Survived'].values == expected['Survived'].values).all()
  assert (actual['Pclass'].values == expected['Pclass'].values).all()


  del mod_orig
  del mod_rewr



def test_ser_simple():
  cell = "x = df['Pclass'].sort_values(ascending=False).head(n=2)"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert (actual.values == expected.values).all()

  del mod_orig
  del mod_rewr


# def test_ser_fail_precond():
#   cell = "x = df['Name'].sort_values(ascending=False).head(n=2)"
#   mod_orig, mod_rewr = common.boiler(cell)

#   actual = mod_orig.__dict__['x']
#   expected = mod_rewr.__dict__['x']

#   assert actual.equals(expected)

#   del mod_orig
#   del mod_rewr