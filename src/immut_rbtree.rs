use std::borrow::BorrowFrom;
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less, Equal};
use std::rc::{Rc};
use std;
use std::iter;
use std::default::Default;

use super::ImmutSList;
use self::ImmutRbTree_::*;
use self::Color::*;

// Okasaki-style persistent red-black trees, with Matt
// Might-style deletion, based on Stephanie Weirich's Haskell
// implementation.

macro_rules! opt_or {
    ($e:expr) => { $e };
    ($e:expr, $($rest:expr),+) => {
        match $e {
            None => opt_or!($($rest),+),
            some => some
        }
    }
}

pub struct ImmutRbTree<K,V> {
    tree: ImmutRbTree_<K,V>
}

pub struct Entry<K,V> {
    k: K,
    v: V
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Dir { Left, Right }
impl Dir {
    pub fn turnaround(self) -> Dir {
        if self == Dir::Left { Dir::Right } else { Dir::Left }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Order { In, Pre, Post }

pub struct Zipper<'a, K:'a, V:'a> {
    zipper: Z<'a, K, V>
}

pub struct Items<'a, K: 'a, V:'a> 
{
    front: Option<Zipper<'a, K, V>>,
    back: Option<Zipper<'a, K, V>>,
    order: Order,
    remaining: usize
}

impl<'a, K:Ord, V> Items<'a, K, V> {
    fn next_item(&mut self, t: Order, d: Dir) -> Option<(&'a K, &'a V)> {
        if self.remaining == 0 {
            None
        } else {
            let (entry, next) = match if d == Dir::Right { &(*self).front } else { &(*self).back } {
                &None => (None, None),
                &Some(ref z) => {
                    self.remaining -= 1;
                    (z.entry(), match t {
                        Order::In => z.inorder(d),
                        Order::Pre => z.preorder(d),
                        _ => z.postorder(d)
                    })
                }
            };
            if d == Dir::Right {
                self.front = next;
            } else {
                self.back = next;
            }
            entry
        }
    }
}

pub type Keys<'a, K, V> =
    iter::Map<(&'a K, &'a V), &'a K, Items<'a, K, V>, fn((&'a K, &'a V)) -> &'a K>;

pub type Values<'a, K, V> =
    iter::Map<(&'a K, &'a V), &'a V, Items<'a, K, V>, fn((&'a K, &'a V)) -> &'a V>;

fn snd<A,B>((_, b): (A, B)) -> B { b }
fn fst<A,B>((a, _): (A, B)) -> A { a }

fn mapfst<'a, K:Ord, V>(i: Items<'a, K, V>) -> Keys<'a, K, V> {
    i.map(fst as fn((&'a K, &'a V)) -> &'a K)
}
fn mapsnd<'a, K:Ord, V>(i: Items<'a, K, V>) -> Values<'a, K, V> {
    i.map(snd as fn((&'a K, &'a V)) -> &'a V)
}

impl<'a, K:Ord, V> Iterator for Items<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        let o = self.order;
        self.next_item(o, Dir::Right)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K:Ord, V> DoubleEndedIterator for Items<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> {
        let o = self.order;
        self.next_item(o, Dir::Left)
    }
}

impl <K:Ord,V> ImmutRbTree<K,V> {
    #[inline]
    pub fn new() -> ImmutRbTree<K,V> {
        ImmutRbTree { tree: ImmutRbTree_::new() }
    }

    #[inline]
    pub fn zipper<'a>(&'a self) -> Zipper<'a, K, V> {
        Zipper::new(&self.tree)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
        where Q: BorrowFrom<K> + Ord {
        self.tree.get(key)
    }

    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
        where Q: BorrowFrom<K> + Ord {
        self.tree.contains_key(key)
    }

    #[inline]
    pub fn insert(&self, k: K, v: V) -> ImmutRbTree<K,V> {
        ImmutRbTree { tree: self.tree.insert(k, v) }
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&self, k: &Q) -> ImmutRbTree<K,V> 
        where Q: BorrowFrom<K> + Ord {
        ImmutRbTree { tree: self.tree.remove(k) }
    }
    
    #[inline]
    pub fn max_entry(&self) -> Option<Rc<Entry<K, V>>> {
        self.tree.max_entry()
    }

    #[inline]
    pub fn iter<'a>(&'a self) -> Items<'a, K, V> {
        let front = self.zipper();
        let back = self.zipper();
        Items {
            remaining: self.len(),
            order: Order::In,
            front: front.leftmost_child().or(Some(front)),
            back: back.rightmost_child().or(Some(back))
        }
    }

    #[inline]
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        mapsnd(self.iter())
    }

    #[inline]
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        mapfst(self.iter())
    }

    #[inline]
    pub fn iter_pre<'a>(&'a self) -> Items<'a, K, V> {
        Items {
            remaining: self.len(),
            order: Order::Pre,
            front: Some(self.zipper()),
            back: Some(self.zipper().preorder_end())
        }
    }

    #[inline]
    pub fn keys_pre<'a>(&'a self) -> Keys<'a, K, V> {
        mapfst(self.iter_pre())
    }

    #[inline]
    pub fn values_pre<'a>(&'a self) -> Values<'a, K, V> {
        mapsnd(self.iter_pre())
    }

    #[inline]
    pub fn iter_post<'a>(&'a self) -> Items<'a, K, V> {
        Items {
            remaining: self.len(),
            order: Order::Post,
            front: Some(self.zipper().postorder_start()),
            back: Some(self.zipper())
        }
    }

    #[inline]
    pub fn keys_post<'a>(&'a self) -> Keys<'a, K, V> {
        mapfst(self.iter_post())
    }

    #[inline]
    pub fn values_post<'a>(&'a self) -> Values<'a, K, V> {
        mapsnd(self.iter_post())
    }
}

impl<K:Ord,V> iter::FromIterator<(K,V)> for ImmutRbTree<K,V> {
    fn from_iter<I: Iterator<Item=(K,V)>>(mut iterator: I) -> ImmutRbTree<K,V> {
        let mut m = ImmutRbTree::new();
        for (k,v) in iterator {
            m = m.insert(k,v);
        }
        m
    }
}

impl<K:Ord+std::fmt::String, V:std::fmt::String> std::fmt::String for ImmutRbTree<K,V> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        try!(write!(f, "{{"));
        for (i,(k,v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {}", k, v));
        }
        write!(f, "}}")
    }
}
impl<K:Ord+std::fmt::String, V:std::fmt::String> std::fmt::Show for ImmutRbTree<K,V> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        try!(write!(f, "{{"));
        for (i,(k,v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {}", k, v));
        }
        write!(f, "}}")
    }
}


impl<K:PartialEq+Ord, V: PartialEq> PartialEq for ImmutRbTree<K, V> {
    fn eq(&self, other: &ImmutRbTree<K, V>) -> bool {
        self.len() == other.len() && 
            self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
impl<K: Eq + Ord, V: Eq> Eq for ImmutRbTree<K, V> {}

impl<K: Ord, V: PartialOrd> PartialOrd for ImmutRbTree<K, V> {
    #[inline]
    fn partial_cmp(&self, other: &ImmutRbTree<K,V>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

impl<K: Ord, V: Ord> Ord for ImmutRbTree<K, V> {
    #[inline]
    fn cmp(&self, other: &ImmutRbTree<K,V>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

impl<K: Ord, V> Default for ImmutRbTree<K,V> {
    #[inline]
    fn default() -> ImmutRbTree<K,V> { ImmutRbTree::new() }
}

#[inline]
fn tozipper<'a, K, V>(z: Z<'a, K, V>) -> Zipper<'a, K, V> {
    Zipper { zipper: z }
}

impl<'a, K:Ord, V> Zipper<'a, K, V> {
    #[inline]
    fn new(tree: &'a ImmutRbTree_<K,V>) -> Zipper<'a, K, V> {
        Zipper { zipper: Z::new(tree) }
    }

    #[inline]
    pub fn entry(&self) -> Option<(&'a K, &'a V)> {
        self.zipper.entry()
    }

    #[inline]
    pub fn reroot(&self) -> Zipper<'a, K, V> {
        tozipper(self.zipper.reroot())
    }

    #[inline]
    pub fn up(&self) -> Option<(Zipper<'a, K, V>, Dir)> {
        self.zipper.up().map(|pair| (tozipper(pair.0), pair.1))
    }

    #[inline]
    pub fn right(&self) -> Option<Zipper<'a, K, V>> {
        self.zipper.right().map(tozipper)            
    }

    #[inline]
    pub fn left(&self) -> Option<Zipper<'a, K, V>> {
        self.zipper.left().map(tozipper)
    }
    
    #[inline]
    pub fn inorder(&self, d: Dir) -> Option<Zipper<'a, K, V>> {
        self.zipper.inorder(d).map(tozipper)
    }

    #[inline]
    pub fn preorder(&self, d: Dir) -> Option<Zipper<'a, K, V>> {
        self.zipper.preorder(d).map(tozipper)
    }

    #[inline]
    pub fn postorder(&self, d: Dir) -> Option<Zipper<'a, K, V>> {
        self.zipper.postorder(d).map(tozipper)
    }

    #[inline]
    pub fn postorder_start(&self) -> Zipper<'a, K, V> {
        tozipper(self.zipper.postorder_start())
    }

    #[inline]
    pub fn preorder_end(&self) -> Zipper<'a, K, V> {
        tozipper(self.zipper.preorder_end())
    }

    #[inline]
    pub fn rightmost_child(&self) -> Option<Zipper<'a, K, V>> {
        self.zipper.dirmost_child(Dir::Right).map(tozipper)
    }

    #[inline]
    pub fn leftmost_child(&self) -> Option<Zipper<'a, K, V>> {
        self.zipper.dirmost_child(Dir::Left).map(tozipper)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Show)]
enum Color {
    R, B, BB, NB
}

impl Color {
    pub fn redder(&self) -> Color {
        match *self {
            BB => B,
            B => R,
            R => NB,
            NB => panic!("Can't make negative black more red!")
        }
    }

    pub fn blacker(&self) -> Color {
        match *self {
            NB => R,
            R => B,
            B => BB,
            BB => panic!("Can't make double black more black!")
        }
    }
}

enum ImmutRbTree_<K,V> {
    BlackLeaf,
    DoubleBlackLeaf,
    Node(Color,Rc<ImmutRbTree_<K,V>>,Rc<Entry<K,V>>,Rc<ImmutRbTree_<K,V>>,usize)
}

// using "derive(Clone)" places spurious Clone constraings on K and V.
impl<K,V> Clone for ImmutRbTree_<K,V> {
    fn clone(&self) -> ImmutRbTree_<K,V> {
        match *self {
            BlackLeaf => BlackLeaf,
            DoubleBlackLeaf => DoubleBlackLeaf,
            Node(ref c, ref l, ref e, ref r, len) => Node(c.clone(), l.clone(), e.clone(),
                                                          r.clone(), len)
        }
    }
}

// actual implementation
impl<K:Ord,V> ImmutRbTree_<K,V> {
    #[inline]
    fn new() -> ImmutRbTree_<K,V> {
        BlackLeaf
    }
    
    #[inline]
    fn len(&self) -> usize {
        match *self {
            Node(_,_,_,_,size) => size,
            _ => 0
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        match *self {
            BlackLeaf|DoubleBlackLeaf => true,
            _ => false
        }
    }

    fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V> 
        where Q: BorrowFrom<K> + Ord
    {
        match *self {
            BlackLeaf => None,
            DoubleBlackLeaf => None,
            Node(_, ref left, ref entry, ref right, _) => {
                match key.cmp(BorrowFrom::borrow_from(&entry.k)) {
                    Less => left.get(key),
                    Greater => right.get(key),
                    Equal => Some(&entry.v)
                }
            }
        }
    }

    #[inline]
    fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool 
        where Q: BorrowFrom<K> + Ord
    {
        self.get(key).is_some()
    }

    fn insert(&self, k: K, v: V) -> ImmutRbTree_<K,V> {
        self.insert_helper(k, v).blacken_insert()
    }

    fn remove<Q: ?Sized>(&self, k: &Q) -> ImmutRbTree_<K,V> 
        where Q: BorrowFrom<K> + Ord
    {
        self.remove_helper(k).blacken_remove()
    }

    fn max_entry(&self) -> Option<Rc<Entry<K,V>>> {
        match *self {
            BlackLeaf|DoubleBlackLeaf => None,
            Node(_, _, ref e, ref l, _) => {
                match **l {
                    DoubleBlackLeaf => panic!("Unexpected double black leaf in max_entry!"),
                    BlackLeaf => Some(e.clone()),
                    ref l => l.max_entry()
                }
            }
        }
    }

    // Private helpers. Don't be alarmed by the plenitude of panic!()s!

    fn redden(&self) -> ImmutRbTree_<K,V> {
        match *self {
            DoubleBlackLeaf => DoubleBlackLeaf,
            BlackLeaf => BlackLeaf,
            Node(_, ref left, ref entry, ref right, len) => {
                Node(R, left.clone(), entry.clone(), right.clone(), len)
            }
        }
    }

    // after an insertion, make the root black unconditionally.
    fn blacken_insert(&self) -> ImmutRbTree_<K,V> {
        match *self {
            Node(B, ref left, ref entry, ref right, len)|
            Node(R, ref left, ref entry, ref right, len) => {
                Node(B, left.clone(), entry.clone(), right.clone(), len)
            }
            _ => panic!("Unexpected color or unexpected leaf on blacken_insert!")
        }
    }

    fn insert_helper(&self, k: K, v: V) -> ImmutRbTree_<K,V> {
        match *self {
            BlackLeaf => {
                let e = Rc::new(BlackLeaf);
                Node(R, e.clone(), Rc::new(Entry{k: k, v: v}), e, 1)
            },
            DoubleBlackLeaf => panic!("Unexpected color in insert_helper!"),
            Node(ref color, ref l, ref e, ref r, ref sz) => {
                match k.cmp(&e.k) {
                    Less => balance(*color, &Rc::new(l.insert_helper(k, v)), e, r),
                    Greater => balance(*color, l, e, &Rc::new(r.insert_helper(k, v))),
                    Equal => Node(*color, l.clone(), 
                                  Rc::new(Entry {k: k, v: v}), r.clone(), *sz) 
                }
            }
        }
    }

    fn remove_helper<Q: ?Sized>(&self, k: &Q) -> ImmutRbTree_<K,V> 
        where Q: BorrowFrom<K> + Ord
    {
        match *self {
            DoubleBlackLeaf => panic!("invalid color in delete_helper!"),
            BlackLeaf => BlackLeaf,
            Node(ref color, ref left, ref e, ref right, _) => {                
                match k.cmp(BorrowFrom::borrow_from(&e.k)) {
                    Equal => self.delete(),
                    Less => bubble(color.clone(), &Rc::new(left.remove_helper(k)), e, right),
                    Greater => bubble(color.clone(), left, e, &Rc::new(right.remove_helper(k)))
                }
            }
        }
    }

    fn delete_max(&self) -> ImmutRbTree_<K,V> {
        match *self {
            // see comment in remove(). The only time this is called
            // we know that there's a maximum.
            BlackLeaf|DoubleBlackLeaf => panic!("No maximum to remove!"),
            Node(ref color, ref left, ref e, ref right, _) => {
                if right.is_empty() {
                    // this may result in a double-black; if so, it
                    // will be caught by the bubble call in remove
                    // after the call to remove_max.
                    self.delete()
                } else {
                    bubble(color.clone(), left, e, &Rc::new(right.delete_max()))
                }
            }
        }
    }

    // Remove myself, returning a new tree. It is possible for the new
    // tree to be a double black leaf; this and bubble() are the only
    // places where double-blackness is introduced, and this is the
    // only place where double-blackness is introduced explicitly, in
    // the form of a double-black leaf.
    fn delete(&self) -> ImmutRbTree_<K,V> {
        if let Node(ref color, ref left, _, ref right, _) = *self {
            if left.is_empty() && right.is_empty() {
                match *color {
                    // since leaves are black, deleting this subtree
                    // and replacing it with a single black leaf
                    // preserves the black-count invariant.
                    R => return BlackLeaf,
                    // a black node with black leaves; replace with a
                    // double-black leaf.
                    B => return DoubleBlackLeaf, 
                    _ => panic!("Impossible color!")
                }
            } else if *color == B {
                // I am black and my sole non-leaf child is red;
                // recolor that child to black.
                // (we couldn't have a black node with one black child:
                //
                //        B1
                //       /  \
                //      L    B2
                //          /  \
                //         T    T
                //
                // (where "L" is a leaf and "T" is a subtree). Since L
                // counts as black, the count of blacks through B1 to
                // L is 1 + the count of blacks to B1. But this must
                // be less than the count of blacks through B2 to any
                // leaf, since T is either a leaf (in which case the
                // count is 2 + B1), or a black or red node (in which
                // case the count is at least 2 + B1). For similar
                // reasons we can't have a red node with a single
                // black child.
                if left.is_empty() {
                    if let Node(R, ref a, ref x, ref b, _) = **right {
                        return fromparts(B, a, x, b)
                    }
                } else if right.is_empty() {
                    if let Node(R, ref a, ref x, ref b, _) = **left {
                        return fromparts(B, a, x, b)
                    }
                }
            }
            // We're removing a node with two children. Find the
            // largest of the entries that's smaller than the current
            // entry, remove that node (which will fall under one of
            // the cases above, since its right node will be empty),
            // and reconstruct the tree at this point with that entry.
            //
            // By the reasoning given just above, by the time we get
            // here we know that left cannot be empty. The
            // possibilities for right are {empty, red, black}; the
            // first two, given that left is empty, are handled by one
            // of the branches above, and the third violates the RB
            // invariants. Consequently it must have a maximum entry,
            // so unwrap() is ok.
            let leftmax = &left.max_entry().unwrap();
            let new_left = &Rc::new(left.delete_max());
            return bubble(color.clone(), new_left, leftmax, right)
        }
        panic!("Impossible case in delete!")
    }

    fn is_double_black(&self) -> bool {
        match *self {
            DoubleBlackLeaf => true,
            Node(BB,_,_,_,_) => true,
            _ => false
        }
    }        

    // after deletion, we may have propagated a double black all the
    // way to the root. In that case, just make it black.
    fn blacken_remove(&self) -> ImmutRbTree_<K,V> {
        match *self {
            BlackLeaf|DoubleBlackLeaf => BlackLeaf,
            Node(B, ref left, ref entry, ref right, len)|
            Node(BB, ref left, ref entry, ref right, len) => {
                Node(B, left.clone(), entry.clone(), right.clone(), len)
            },
            _ => panic!("Unexpected color on blacken_delete!")
        }
    }

    fn redder(&self) -> ImmutRbTree_<K,V> {
        match *self {
            DoubleBlackLeaf => BlackLeaf,
            BlackLeaf => panic!("Can't make black leaf redder!"),
            Node(ref color, ref left, ref entry, ref right, len) => {
                Node(color.redder(), left.clone(), entry.clone(), right.clone(), len)
            }
        }
    }
}

#[inline]
fn fromparts<K:Ord,V>(c: Color,
             left: &Rc<ImmutRbTree_<K,V>>, 
             e: &Rc<Entry<K,V>>,
             right: &Rc<ImmutRbTree_<K,V>>) -> ImmutRbTree_<K,V> {
    Node(c.clone(), left.clone(), e.clone(), right.clone(), left.len() + (&*right).len() + 1u)
}

#[inline]
fn bubble<K:Ord,V>(c: Color,
                 left: &Rc<ImmutRbTree_<K,V>>,
                 e: &Rc<Entry<K,V>>,
                 right: &Rc<ImmutRbTree_<K,V>>) -> ImmutRbTree_<K,V> {
    if left.is_double_black() || right.is_double_black() {
        // make a tree whose root is one shade blacker and whose
        // children are one shade redder. Since double-blackness is
        // introduced only at one leaf by deletion, and is propagated
        // upwards by this function, we can be certain that this
        // maintains the correct blackness count, and that we can make
        // c blacker (it must be either red or black on entry, and
        // will be either black or double-black on exit). Since this
        // will create either a red child or a negative-black child,
        // we have to balance.
        balance(c.blacker(), &Rc::new(left.redder()), e, &Rc::new(right.redder()))
    } else {
        balance(c, left, e, right)
    }
}

fn balance_black_colors<K:Ord,V>(is_double: bool,
                                 left: &Rc<ImmutRbTree_<K,V>>,
                                 e: &Rc<Entry<K,V>>,
                                 right: &Rc<ImmutRbTree_<K,V>>) -> ImmutRbTree_<K,V> {
    // the root joining left and right is either double-black
    // (is_double is true) or black. If it has a red child with a red
    // child, then we make the child (or grandchild) with the "middle"
    // value the new root, color both its children black, and color
    // the new root either red (old root was black) or black (old root
    // was double-black).
    let outer_color = if is_double { B } else { R };
    if let Node(R, ref ll, ref le, ref lr, _) = **left {
        if let Node(R, ref a, ref x, ref b, _) = **ll {
            return fromparts(outer_color,
                             &Rc::new(fromparts(B, a, x, b)),
                             le,
                             &Rc::new(fromparts(B, lr, e, right)))
        } else if let Node(R, ref b, ref y, ref c, _) = **lr {
            return fromparts(outer_color,
                             &Rc::new(fromparts(B, ll, le, b)),
                             y, 
                             &Rc::new(fromparts(B, c, e, right)))
        }
    }
    if let Node(R, ref rl, ref re, ref rr, _) = **right {
        if let Node(R, ref b, ref y, ref c, _) = **rl {
            return fromparts(outer_color,
                             &Rc::new(fromparts(B, left, e, b)),
                             y,
                             &Rc::new(fromparts(B, c, re, rr)))
        } else if let Node(R, ref c, ref z, ref d, _) = **rr {
            return fromparts(outer_color,
                             &Rc::new(fromparts(B, left, e, rl)),
                             re,
                             &Rc::new(fromparts(B, c, z, d)))
        }
    }
    // this branch is only taken if we were deleting, in which case we
    // might have a negative-black child. Negative blacks are created
    // when a red node is made redder in bubble(); consequently they
    // have black, non-leaf children. (In more detail: a negative
    // black is created when a double black has a red sibling. Red
    // nodes can't have only one leaf-child, so we would have to be in
    // this situation prior to bubbling:
    //
    //              B 
    //            /   \
    //           R     DB
    //          / \   /  \
    //         L   L T    T
    //
    // But there's no way for this to conform to the RB tree
    // invariants; the double-black matches the two leaves of the red
    // node, but each of its sub-trees will contribute more.
    // So we must have something like this by the time we get here:
    //
    //              DB1
    //            /    \
    //           NB2    B5
    //          /  \    
    //         B3   B4
    //             /  \ 
    //            T6  T7
    //
    // which we can rewrite like this:
    //
    //           B4
    //         /    \
    //        B2     B1
    //       /  \    / \
    //      R3  T6 T7   B5
    //
    // We may have to rebalance at the newly created red node, but
    // this has eliminated both the negative black and the double black.
    if is_double {
        if let Node(NB, ref ll, ref le, ref lr, _) = **left {
            if let Node(B, ref b, ref y, ref c, _) = **lr {
                if let ref a@Node(B, _, _, _, _) = **ll {
                    let new_right = Rc::new(fromparts(B, c, e, right));
                    let new_left = Rc::new(balance(B, &Rc::new(a.redden()), le, b));
                    return fromparts(B, &new_left, y, &new_right)
                }
            }
        }
        if let Node(NB, ref rl, ref z, ref rr, _) = **right {
            if let Node(B, ref b, ref y, ref c, _) = **rl {
                if let ref d@Node(B, _, _, _, _) = **rr {
                    let new_right = Rc::new(fromparts(B, left, e, b));
                    let new_left = Rc::new(balance(B, c, z, &Rc::new(d.redden())));
                    return fromparts(B, &new_right, y, &new_left)
                }
            }
        }
    }
    // nothing interesting, so just create a new tree, possibly with a
    // double-black at the root.
    fromparts(if is_double { BB } else { B }, left, e, right)
}

fn balance<K:Ord,V>(c: Color,
                    left: &Rc<ImmutRbTree_<K,V>>,
                    e: &Rc<Entry<K,V>>,
                    right: &Rc<ImmutRbTree_<K,V>>) -> ImmutRbTree_<K,V> {
    match c {
        B|BB => balance_black_colors(c == BB, left, e, right),
        c => fromparts(c, left, e, right)
    }
}

struct Z<'a, K:'a, V:'a> {
    context: ImmutSList<(Dir, &'a ImmutRbTree_<K,V>)>,
    node: &'a ImmutRbTree_<K,V>
}

impl<'a,K:Ord,V> Z<'a,K,V> {
    #[inline]
    fn new(tree: &'a ImmutRbTree_<K,V>) -> Z<'a, K, V> {
        Z { context: ImmutSList::new(), node: tree }
    }

    #[inline]
    fn reroot(&self) -> Z<'a, K, V> {
        Z { context: ImmutSList::new(), node: self.node }
    }

    #[inline]
    fn entry(&self) -> Option<(&'a K, &'a V)> {
        match *self.node {
            BlackLeaf | DoubleBlackLeaf => None,
            Node(_, _, ref entry, _, _) => Some((&entry.k, &entry.v))
        }
    }

    #[inline]
    fn up(&self) -> Option<(Z<'a, K, V>, Dir)> {
        match self.context.head() {
            None => None,
            Some(&(d, parent)) => {
                Some((Z { context: self.context.tail(), node: parent }, d))
            }
        }
    }

    fn inorder(&self, dir: Dir) -> Option<Z<'a, K, V>> {
        match self.go_dir(dir) {
            None => match self.up() {
                None => None,
                Some((z, prev_dir)) => {
                    let mut prev_dir = prev_dir;
                    let mut z = z;
                    loop {
                        if prev_dir != dir {
                            return Some(z)
                        } else {
                            match z.up() {
                                None => return None,
                                Some((zz, d)) => {
                                    prev_dir = d;
                                    z = zz;
                                }
                            }
                        }
                    }
                }
            },
            Some(d) => d.dirmost_child(dir.turnaround()).or(Some(d))
        }
    }

    fn preorder(&self, dir: Dir) -> Option<Z<'a, K, V>> {
        if dir == Dir::Right {
            match self.left() {
                None => {
                    match self.right() {
                        None => {
                            let mut u = self.up();
                            loop {
                                match u {
                                    None => return None,
                                    Some((uu, Dir::Right)) => u = uu.up(),
                                    Some((uu, _)) => {
                                        if uu.can_go(Dir::Right) {
                                            return uu.right()
                                        } else { u = uu.up() }
                                    }
                                }
                            }
                        },
                        some => some
                    }
                },
                some => some
            }
        } else {
            match self.up() {
                None => None,
                Some((u,Dir::Left)) => Some(u),
                Some((u,_)) => {
                    let mut z = u;
                    loop {
                        match z.left() {
                            None => return Some(z),
                            Some(l) => {
                                z = l;
                                while let Some(r) = z.right() {
                                    z = r
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn preorder_end(&self) -> Z<'a, K, V> {
        match self.dirmost_child(Dir::Right) {
            None => Z { context: self.context.clone(), node: self.node },
            Some(r) => {
                let mut z = r;
                loop {
                    match z.left() {
                        None => return z,
                        Some(l) => match l.dirmost_child(Dir::Right) {
                            Some(r) => z = r,
                            None => return l
                        }
                    }
                }
            }
        }
    }

    fn postorder_start(&self) -> Z<'a, K, V> {
        let mut start = opt_or!(self.dirmost_child(Dir::Left), self.right(), Some(Z { context: self.context.clone(), node: self.node})).unwrap();
        loop {
            match start.dirmost_child(Dir::Left) {
                Some(c) => start = c,
                None => match start.right() {
                    None => return start,
                    Some(c) => start = c
                }
            }
        }
    }

    fn postorder(&self, d: Dir) -> Option<Z<'a, K, V>> {
        if d == Dir::Right {
            match self.up() {
                None => None,
                Some((u, Dir::Right)) => Some(u),
                Some((z, _)) => {
                    let mut z = z;
                    loop {
                        match z.right() {
                            None => return Some(z),
                            Some(r) => {
                                z = r;
                                while let Some(l) = z.left() {
                                    z = l
                                }
                            }
                        }
                    }
                }
            }
        } else {
            match self.right() {
                None => { 
                    match self.left() {
                        None => {
                            let mut u = self.up();
                            loop {
                                match u {
                                    None => return None,
                                    Some((uu, Dir::Left)) => u = uu.up(),
                                    Some((uu, _)) => {
                                        if uu.can_go(Dir::Left) {
                                            return uu.left()
                                        } else { u = uu.up() }
                                        // the below, which should be
                                        // the same as the above in
                                        // effect, caused illegal instructions.
                                        // match uu.left() {
                                        //     None => u = uu.up(),
                                        //     some => return some
                                        // }
                                    }
                                }
                            }
                        },
                        some => some
                    }
                }
                some => some
            }
        }
    }


    fn right(&self) -> Option<Z<'a, K, V>> {
        match *self.node {
            BlackLeaf | DoubleBlackLeaf => None,
            Node(_, _, _, ref right, _) => if right.is_empty() {
                None
            } else {
                Some(Z { context: self.context.append((Dir::Right, self.node)),
                         node: &**right })
            }
        }
    }

    fn can_go(&self, dir: Dir) -> bool {
        match *self.node {
            BlackLeaf | DoubleBlackLeaf => false,
            Node(_, ref left, _, ref right, _) =>
                (dir == Dir::Left && !left.is_empty()) ||
                (dir == Dir::Right && !right.is_empty())
        }
    }

    fn go_dir(&self, dir: Dir) -> Option<Z<'a, K, V>> {
        match dir {
            Dir::Right => self.right(),
            Dir::Left => self.left()
        }
    }

    fn dirmost_child(&self, dir: Dir) -> Option<Z<'a, K, V>> {
        if !self.can_go(dir) {
            None
        } else {
            let mut next = self.go_dir(dir).unwrap();
            loop {
                let nnext = next.go_dir(dir);
                match nnext {
                    None => return Some(next),
                    Some(z) => next = z
                }
            }
        }
    }

    fn left(&self) -> Option<Z<'a, K, V>> {
        match *self.node {
            BlackLeaf | DoubleBlackLeaf => None,
            Node(_, ref left, _, _, _) => if left.is_empty() {
                None
            } else {
                Some(Z { context: self.context.append((Dir::Left, self.node)),
                         node: &**left })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ImmutRbTree;
    use super::ImmutRbTree_::*;
    use super::Color::*;
    use super::Color;
    use super::ImmutRbTree_;
    use std::rand::Rng;
    use std::rand;
    
    macro_rules! opt_eq {
        ($e:expr, $t:expr) => {
            {
                let e = $e;
                let t = $t;
                assert!(e.is_some());
                assert!(e.unwrap() == t)
            }
        }
    }
    fn has<K:Ord,V:PartialEq>(m: &ImmutRbTree<K,V>, k: &K, v: &V) {
        opt_eq!(m.get(k), v)
    }

    fn hasnt<K:Ord, V>(m: &ImmutRbTree<K,V>, k: K) {
        assert!(m.get(&k).is_none())
    }

    fn hasall<K:Ord,V:PartialEq>(m: &ImmutRbTree<K,V>, kvs: &[(K,V)]) {
        for &(ref k,ref v) in kvs.iter() {
            has(m, k, v)
        }
    }

    #[test]
    fn test_basic() {
        let m: ImmutRbTree<int, int> = ImmutRbTree::new();
        let m = m.insert(1, 2);
        let m = m.insert(4, 5);
        let m = m.insert(-1, 4);
        let m = m.insert(3, 3);
        assert_eq!(m.len(), 4);
        has(&m, &1, &2);
        let m2 = m.remove(&1);
        assert_eq!(m2.len(), 3);
        hasnt(&m2, 1);
        has(&m, &1, &2);
        hasall(&m, vec![(4,5),(-1,4), (3,3)].as_slice());
    }

    fn paired(v: &[int]) -> Vec<(int, int)> {
        v.iter().zip(v.iter()).map(|(x,y)| (*x,*y)).collect()
    }

    fn gen_map(v: &[int]) -> ImmutRbTree<int, int> {
        paired(v).iter().map(|&x| x).collect()
    }

    #[test]
    fn test_from_iter() {
        let v = &[0,1,2,3,4,5,6];
        let m = gen_map(v);
        hasall(&m, paired(v).as_slice());
        hasnt(&m, -1); hasnt(&m, 7);
        assert_eq!(m.len(), 7);
    }

    fn pair_deref(p: (&int, &int)) -> (int,int) {
        (*p.0, *p.1)
    }

    #[test]
    fn test_orders() {
        let v = &[-2,0,2,3,4,7,9];
        let m = gen_map(v);
        let m1 = m.insert(-3,-3).insert(10,10);
        let m2 = m.insert(1,1).insert(5,5);

        /*
          m1 is shaped like (keys only):

                       3
                     /   \
                    0     7
                   / \   / \
                 -2   2 4   9
                 /           \
               -3             10

          m2 is shaped like:

                       3
                     /   \
                    0     7
                   / \   / \
                 -2   2 4   9
                     /   \     
                    1     5
        */
        
        let mut x : Vec<(int, int)> = m1.iter().map(pair_deref).collect();
        assert_eq!(x, vec![(-3, -3), (-2, -2), (0, 0), (2, 2), (3, 3),
                           (4, 4), (7, 7), (9, 9), (10, 10)]);
        x = m1.iter().rev().map(pair_deref).collect();
        assert_eq!(x, vec![(10,10), (9,9), (7,7), (4,4), (3,3), 
                           (2,2), (0,0), (-2, -2), (-3, -3)]);

        x = m2.iter().map(pair_deref).collect();
        assert_eq!(x, vec![(-2, -2), (0, 0), (1, 1), (2, 2), (3, 3),
                           (4, 4), (5,5), (7, 7), (9, 9)]);
        x = m2.iter().rev().map(pair_deref).collect();
        assert_eq!(x, vec![(9,9), (7,7), (5,5), (4,4), (3,3),
                           (2,2), (1,1), (0,0), (-2,-2)]);

        x = m1.iter_pre().map(pair_deref).collect();
        assert_eq!(x, vec![(3, 3), (0, 0), (-2, -2), (-3, -3),
                           (2, 2), (7, 7), (4, 4), (9, 9), (10, 10)]);
        x = m1.iter_pre().rev().map(pair_deref).collect();
        assert_eq!(x, vec![(10, 10), (9, 9), (4, 4), (7, 7), (2, 2),
                           (-3, -3), (-2, -2), (0, 0), (3, 3)]);

        x = m2.iter_pre().map(pair_deref).collect();
        assert_eq!(x, vec![(3,3), (0,0), (-2, -2), (2,2), (1,1), (7,7),
                           (4,4), (5,5), (9,9)]);
        x = m2.iter_pre().rev().map(pair_deref).collect();
        assert_eq!(x, vec![(9,9), (5,5), (4,4), (7,7), (1,1), (2,2),
                           (-2,-2), (0,0), (3,3)]);
        
        x = m1.iter_post().map(pair_deref).collect();
        assert_eq!(x, vec![(-3, -3), (-2, -2), (2, 2), (0, 0), (4, 4),
                           (10, 10), (9, 9), (7, 7), (3, 3)]);
        x = m1.iter_post().rev().map(pair_deref).collect();
        assert_eq!(x, vec![(3, 3), (7, 7), (9, 9), (10, 10), (4, 4),
                           (0, 0), (2, 2), (-2, -2), (-3, -3)]);

        x = m2.iter_post().map(pair_deref).collect();
        assert_eq!(x, vec![(-2,-2), (1,1), (2,2), (0,0), (5,5), 
                           (4,4), (9,9), (7,7), (3,3)]);
        x = m2.iter_post().rev().map(pair_deref).collect();
        assert_eq!(x, vec![(3,3), (7,7), (9,9), (4,4), (5,5), (0,0),
                           (2,2), (1,1), (-2,-2)]);
    }

    #[test]
    fn test_pop() {
        let m = ImmutRbTree::new().insert(1i,1i);
        assert_eq!(m.remove(&3), m);
        assert_eq!(m.insert(2,2).remove(&2), m);
    }

    #[test]
    fn test_format() {
        let mut m : ImmutRbTree<int, int> = ImmutRbTree::new();
        assert!(format!("{}", m) == "{}");
        m = m.insert(4, 10).insert(8, 9);
        assert!(format!("{}", m) == "{4: 10, 8: 9}");
    }

    #[test]
    fn test_len() {
        let mut m = ImmutRbTree::new();
        m = m.insert(1i, 1i);
        assert_eq!(m.len(), 1);
        assert_eq!(m.insert(1,2).len(), 1);
        assert_eq!(m.insert(2,2).len(), 2);
        m = m.insert(2,2).insert(3,3).insert(4,4);
        assert_eq!(m.len(), 4);
        assert_eq!(m.remove(&3).len(), 3);
        assert_eq!(m.insert(3, 30).len(), 4);
    }

    #[test]
    fn test_ord() {
        let mut a = ImmutRbTree::new();
        let mut b = ImmutRbTree::new();
        assert!(a <= b && b >= a);
        a = a.insert(1i, 2i);
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        b = b.insert(1i, 1i);
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        b = b.remove(&1).insert(2,2);
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_eq() {
        let mut a = ImmutRbTree::new();
        let mut b = ImmutRbTree::new();
        assert!(a  == b);
        a = a.insert(1i, 2i);
        assert!(a != b);
        b = b.insert(1i, 1i);
        assert!(a != b);
        println!("{}", a.insert(1,1));
        assert!(a.insert(1,1) == b);
    }

    #[test]
    fn test_zip_basic() {
        let mut m = ImmutRbTree::new();
        m = m.insert(1i, 2i);
        m = m.insert(3, 5);
        m = m.insert(6, 7);
        m = m.insert(4, 2);
        let z = m.zipper().leftmost_child().unwrap();
        let e = z.entry();
        let v = (&1, &2);
        opt_eq!(e, v);
    }

    fn check_child(t: &ImmutRbTree_<int, int>, parent: Color) -> usize {
        match t {
            &DoubleBlackLeaf => {assert!(false, "Double black leaf!"); 0},
            &BlackLeaf => 1,
            &Node(ref c, ref r, _, ref l, _) => {
                assert!(*c == R || *c == B, "Escaped NB or BB!");
                assert!(!(parent == R && *c == R), "Red parent has red child!");
                let right_path = check_child(&**r, *c);
                let left_path = check_child(&**l, *c);
                assert!(right_path == left_path);
                right_path + (if *c == B { 1 } else { 0 })
            }
        }
    }

    fn check_structure(t: &ImmutRbTree_<int, int>) {
        match t {
            &DoubleBlackLeaf => assert!(false, "double black at root!"),
            &BlackLeaf => (),
            &Node(B, ref r, _, ref l, _) => {
                assert_eq!(check_child(&**r, B), check_child(&**l, B));
            }
            _ => assert!(false, "not rooted at black leaf!")
        }
    }

    #[test]
    fn test_rand_int() {
        let mut m : ImmutRbTree<int, int> = ImmutRbTree::new();
        let seed: &[_] = &[42];
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);
        let mut cnt = 0;
        for _ in range(0u, 3) {
            for _ in range(0u, 90) {
                let k = rng.gen();
                let v = rng.gen();
                if !m.contains_key(&k) {
                    cnt += 1;
                }
                m = m.insert(k, v);
                assert_eq!(m.len(), cnt);
                check_structure(&m.tree);
            }
        }
        for _ in range(0u, 100) {
            let root = m.zipper().entry().map(|(&k,_)| k);
            match root {
                None => (),
                Some(k) => {
                    m = m.remove(&k);
                    check_structure(&m.tree)
                }
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use test::{Bencher, black_box};
    use test;
    use std::rand::{weak_rng, Rng};
    use super::ImmutRbTree;

    #[bench]
    fn bench_insert(b: &mut Bencher) {
        let mut m = ImmutRbTree::new();
        let mut cur = 0i;
        b.iter(|| {
            m = m.insert(cur, cur);
            cur += 1;
        })
    }

    fn bench_iter(b: &mut Bencher, size: usize) {
        let mut m = ImmutRbTree::<usize,usize>::new();
        let mut rng = weak_rng();
        for _ in range(0, size) {
            m = m.insert(rng.gen(), rng.gen())
        };

        b.iter(|| {
            for entry in m.iter() {
                black_box(entry);
            }
        })
    }
    
    #[bench]
    pub fn iter_20(b: &mut Bencher) {
        bench_iter(b, 20);
    }
    
    #[bench]
    pub fn iter_1000(b: &mut Bencher) {
        bench_iter(b, 1000);
    }

    fn insert_rand_n(b: &mut Bencher, n: usize) {
        let mut m = ImmutRbTree::<usize, usize>::new();
        let mut rng = weak_rng();
        for i in range(0, n) {
            m = m.insert(rng.gen::<usize>() % n, i)
        }
        b.iter(|| {
            let k = rng.gen::<usize>() % n;
            m = m.insert(k,k).remove(&k)
        })
    }

    fn insert_seq_n(b: &mut Bencher, n: usize) {
        let mut m = ImmutRbTree::<usize, usize>::new();
        for i in range(0u, n) {
            m = m.insert(i*2, i*2);
        }

        let mut i = 1;
        b.iter(|| {
            m = m.insert(i, i).remove(&i);
            i = (i + 2) % n;
        })
    }

    #[bench]
    pub fn insert_rand_100(b: &mut Bencher) {
        insert_rand_n(b,100);
    }

    #[bench]
    pub fn insert_seq_100(b: &mut Bencher) {
        insert_seq_n(b, 100);
    }

    #[bench]
    pub fn insert_rand_10_000(b: &mut Bencher) {
        insert_rand_n(b, 10_000);
    }

    #[bench]
    pub fn insert_seq_10_000(b: &mut Bencher) {
        insert_seq_n(b, 10_000);
    }
}
