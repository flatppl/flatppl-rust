module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<5.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.5> : tensor<f32>
    %2 = stablehlo.multiply %1, %0 : tensor<f32>
    %3 = stablehlo.constant dense<0.5> : tensor<f32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.compare LT, %2, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = stablehlo.add %2, %5 : tensor<f32>
    %8 = stablehlo.select %6, %7, %2 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %10 = stablehlo.subtract %8, %9 : tensor<f32>
    %11 = stablehlo.constant dense<9.0> : tensor<f32>
    %12 = stablehlo.multiply %11, %10 : tensor<f32>
    %13 = stablehlo.sqrt %12 : tensor<f32>
    %14 = stablehlo.divide %5, %13 : tensor<f32>
    %15, %16 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %17 = stablehlo.constant dense<9> : tensor<128xui32>
    %18 = stablehlo.shift_right_logical %16, %17 : tensor<128xui32>
    %19 = stablehlo.convert %18 : (tensor<128xui32>) -> tensor<128xf32>
    %20 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128xf32>
    %22 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %23 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %24 = stablehlo.multiply %21, %22 : tensor<128xf32>
    %25 = stablehlo.subtract %24, %23 : tensor<128xf32>
    %26 = chlo.erf_inv %25 : tensor<128xf32> -> tensor<128xf32>
    %27 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %28 = stablehlo.multiply %26, %27 : tensor<128xf32>
    %29 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %30 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %31 = stablehlo.multiply %28, %29 : tensor<128xf32>
    %32 = stablehlo.add %31, %30 : tensor<128xf32>
    %33, %34 = stablehlo.rng_bit_generator %15, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %35 = stablehlo.constant dense<9> : tensor<128xui32>
    %36 = stablehlo.shift_right_logical %34, %35 : tensor<128xui32>
    %37 = stablehlo.convert %36 : (tensor<128xui32>) -> tensor<128xf32>
    %38 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %39 = stablehlo.multiply %37, %38 : tensor<128xf32>
    %40 = stablehlo.subtract %5, %4 : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %42 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %43 = stablehlo.multiply %39, %41 : tensor<128xf32>
    %44 = stablehlo.add %43, %42 : tensor<128xf32>
    %45 = stablehlo.constant dense<0> : tensor<i32>
    %46 = stablehlo.constant dense<false> : tensor<i1>
    %47 = stablehlo.constant dense<0.0> : tensor<f32>
    %51:3 = stablehlo.while(%48 = %45, %49 = %46, %50 = %47) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %52 = stablehlo.constant dense<128> : tensor<i32>
      %53 = stablehlo.compare LT, %48, %52, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %54 = stablehlo.not %49 : tensor<i1>
      %55 = stablehlo.and %54, %53 : tensor<i1>
      stablehlo.return %55 : tensor<i1>
    } do {
      %56 = stablehlo.dynamic_slice %32, %48, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %57 = stablehlo.reshape %56 : (tensor<1xf32>) -> tensor<f32>
      %58 = stablehlo.dynamic_slice %44, %48, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %59 = stablehlo.reshape %58 : (tensor<1xf32>) -> tensor<f32>
      %60 = stablehlo.multiply %14, %57 : tensor<f32>
      %61 = stablehlo.add %5, %60 : tensor<f32>
      %62 = stablehlo.multiply %61, %61 : tensor<f32>
      %63 = stablehlo.multiply %62, %61 : tensor<f32>
      %64 = stablehlo.multiply %10, %63 : tensor<f32>
      %65 = stablehlo.constant dense<0.5> : tensor<f32>
      %66 = stablehlo.multiply %57, %57 : tensor<f32>
      %67 = stablehlo.multiply %65, %66 : tensor<f32>
      %68 = stablehlo.multiply %10, %63 : tensor<f32>
      %69 = stablehlo.negate %68 : tensor<f32>
      %70 = stablehlo.log %63 : tensor<f32>
      %71 = stablehlo.multiply %10, %70 : tensor<f32>
      %72 = stablehlo.add %67, %10 : tensor<f32>
      %73 = stablehlo.add %72, %69 : tensor<f32>
      %74 = stablehlo.add %73, %71 : tensor<f32>
      %75 = stablehlo.log %59 : tensor<f32>
      %76 = stablehlo.compare LT, %75, %74 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %77 = stablehlo.compare GT, %63, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %78 = stablehlo.and %76, %77 : tensor<i1>
      %79 = stablehlo.constant dense<1> : tensor<i32>
      %80 = stablehlo.add %48, %79 : tensor<i32>
      stablehlo.return %80, %78, %64 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %81, %82 = stablehlo.rng_bit_generator %33, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %83 = stablehlo.constant dense<9> : tensor<ui32>
    %84 = stablehlo.shift_right_logical %82, %83 : tensor<ui32>
    %85 = stablehlo.convert %84 : (tensor<ui32>) -> tensor<f32>
    %86 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %87 = stablehlo.multiply %85, %86 : tensor<f32>
    %88 = stablehlo.subtract %5, %4 : tensor<f32>
    %89 = stablehlo.multiply %87, %88 : tensor<f32>
    %90 = stablehlo.add %89, %4 : tensor<f32>
    %91 = stablehlo.divide %5, %2 : tensor<f32>
    %92 = stablehlo.power %90, %91 : tensor<f32>
    %93 = stablehlo.select %6, %92, %5 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %94 = stablehlo.multiply %51#2, %93 : tensor<f32>
    %95 = stablehlo.divide %94, %3 : tensor<f32>
    %96 = stablehlo.constant dense<0.0> : tensor<f32>
    %97 = stablehlo.constant dense<1.0> : tensor<f32>
    %98, %99 = stablehlo.rng_bit_generator %81, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %100 = stablehlo.constant dense<9> : tensor<ui32>
    %101 = stablehlo.shift_right_logical %99, %100 : tensor<ui32>
    %102 = stablehlo.convert %101 : (tensor<ui32>) -> tensor<f32>
    %103 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %104 = stablehlo.multiply %102, %103 : tensor<f32>
    %105 = stablehlo.constant dense<2.0> : tensor<f32>
    %106 = stablehlo.constant dense<1.0> : tensor<f32>
    %107 = stablehlo.multiply %104, %105 : tensor<f32>
    %108 = stablehlo.subtract %107, %106 : tensor<f32>
    %109 = chlo.erf_inv %108 : tensor<f32> -> tensor<f32>
    %110 = stablehlo.constant dense<1.4142135> : tensor<f32>
    %111 = stablehlo.multiply %109, %110 : tensor<f32>
    %112 = stablehlo.multiply %111, %97 : tensor<f32>
    %113 = stablehlo.add %112, %96 : tensor<f32>
    %114 = stablehlo.divide %95, %0 : tensor<f32>
    %115 = stablehlo.sqrt %114 : tensor<f32>
    %116 = stablehlo.divide %113, %115 : tensor<f32>
    return %116, %98 : tensor<f32>, tensor<2xui64>
  }
}
