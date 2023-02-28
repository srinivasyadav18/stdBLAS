//  Copyright (c) 2022 Hartmut Kaiser

namespace hpxtesting {

template <class T>
auto create_stdvector_and_copy(T sourceView)
{
    static_assert(sourceView.rank() == 1);

    using value_type = typename T::value_type;
    using res_t = std::vector<value_type>;

    res_t result(sourceView.extent(0));
    for (std::size_t i = 0; i < sourceView.extent(0); ++i)
    {
        result[i] = sourceView(i);
    }

    return result;
}

template<class T>
auto create_stdvector_and_copy_rowwise(T sourceView)
{
  static_assert (sourceView.rank() == 2);

  using value_type = typename T::value_type;
  using res_t = std::vector<value_type>;

  res_t result(sourceView.extent(0)*sourceView.extent(1));
  std::size_t k=0;
  for (std::size_t i=0; i<sourceView.extent(0); ++i){
    for (std::size_t j=0; j<sourceView.extent(1); ++j){
      result[k++] = sourceView(i,j);
    }
  }

  return result;
}

}    // namespace hpxtesting
