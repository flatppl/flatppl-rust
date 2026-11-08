module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<5x3xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<[2.0, 3.0, 4.0]> : tensor<3xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<3xf32>) -> tensor<5x3xf32>
    %2 = stablehlo.reshape %1 : (tensor<5x3xf32>) -> tensor<15xf32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<15xf32>
    %6 = stablehlo.compare LT, %2, %5 : (tensor<15xf32>, tensor<15xf32>) -> tensor<15xi1>
    %7 = stablehlo.add %2, %5 : tensor<15xf32>
    %8 = stablehlo.select %6, %7, %2 : (tensor<15xi1>, tensor<15xf32>, tensor<15xf32>) -> tensor<15xf32>
    %9 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<15xf32>
    %11 = stablehlo.subtract %8, %10 : tensor<15xf32>
    %12 = stablehlo.constant dense<9.0> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<15xf32>
    %14 = stablehlo.multiply %13, %11 : tensor<15xf32>
    %15 = stablehlo.sqrt %14 : tensor<15xf32>
    %16 = stablehlo.divide %5, %15 : tensor<15xf32>
    %17, %18 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x15xui32>)
    %19 = stablehlo.constant dense<9> : tensor<128x15xui32>
    %20 = stablehlo.shift_right_logical %18, %19 : tensor<128x15xui32>
    %21 = stablehlo.convert %20 : (tensor<128x15xui32>) -> tensor<128x15xf32>
    %22 = stablehlo.constant dense<1.1920929E-7> : tensor<128x15xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<128x15xf32>
    %24 = stablehlo.constant dense<2.0> : tensor<128x15xf32>
    %25 = stablehlo.constant dense<1.0> : tensor<128x15xf32>
    %26 = stablehlo.multiply %23, %24 : tensor<128x15xf32>
    %27 = stablehlo.subtract %26, %25 : tensor<128x15xf32>
    %28 = chlo.erf_inv %27 : tensor<128x15xf32> -> tensor<128x15xf32>
    %29 = stablehlo.constant dense<1.4142135623730951> : tensor<128x15xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<128x15xf32>
    %31, %32 = stablehlo.rng_bit_generator %17, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x15xui32>)
    %33 = stablehlo.constant dense<9> : tensor<128x15xui32>
    %34 = stablehlo.shift_right_logical %32, %33 : tensor<128x15xui32>
    %35 = stablehlo.convert %34 : (tensor<128x15xui32>) -> tensor<128x15xf32>
    %36 = stablehlo.constant dense<1.1920929E-7> : tensor<128x15xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<128x15xf32>
    %38 = stablehlo.constant dense<0> : tensor<i32>
    %39 = stablehlo.constant dense<false> : tensor<15xi1>
    %40 = stablehlo.constant dense<0.0> : tensor<15xf32>
    %44:3 = stablehlo.while(%41 = %38, %42 = %39, %43 = %40) : tensor<i32>, tensor<15xi1>, tensor<15xf32>
    cond {
      %45 = stablehlo.constant dense<128> : tensor<i32>
      %46 = stablehlo.compare LT, %41, %45, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %47 = stablehlo.constant dense<true> : tensor<i1>
      %48 = stablehlo.reduce(%42 init: %47) applies stablehlo.and across dimensions = [0] : (tensor<15xi1>, tensor<i1>) -> tensor<i1>
      %49 = stablehlo.not %48 : tensor<i1>
      %50 = stablehlo.and %46, %49 : tensor<i1>
      stablehlo.return %50 : tensor<i1>
    } do {
      %51 = stablehlo.constant dense<0> : tensor<i32>
      %52 = stablehlo.dynamic_slice %30, %41, %51, sizes = [1, 15] : (tensor<128x15xf32>, tensor<i32>, tensor<i32>) -> tensor<1x15xf32>
      %53 = stablehlo.reshape %52 : (tensor<1x15xf32>) -> tensor<15xf32>
      %54 = stablehlo.dynamic_slice %37, %41, %51, sizes = [1, 15] : (tensor<128x15xf32>, tensor<i32>, tensor<i32>) -> tensor<1x15xf32>
      %55 = stablehlo.reshape %54 : (tensor<1x15xf32>) -> tensor<15xf32>
      %56 = stablehlo.multiply %16, %53 : tensor<15xf32>
      %57 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<15xf32>
      %58 = stablehlo.add %57, %56 : tensor<15xf32>
      %59 = stablehlo.multiply %58, %58 : tensor<15xf32>
      %60 = stablehlo.multiply %59, %58 : tensor<15xf32>
      %61 = stablehlo.multiply %11, %60 : tensor<15xf32>
      %62 = stablehlo.constant dense<0.5> : tensor<f32>
      %63 = stablehlo.multiply %53, %53 : tensor<15xf32>
      %64 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<f32>) -> tensor<15xf32>
      %65 = stablehlo.multiply %64, %63 : tensor<15xf32>
      %66 = stablehlo.negate %61 : tensor<15xf32>
      %67 = stablehlo.log %60 : tensor<15xf32>
      %68 = stablehlo.multiply %11, %67 : tensor<15xf32>
      %69 = stablehlo.add %65, %11 : tensor<15xf32>
      %70 = stablehlo.add %69, %66 : tensor<15xf32>
      %71 = stablehlo.add %70, %68 : tensor<15xf32>
      %72 = stablehlo.log %55 : tensor<15xf32>
      %73 = stablehlo.compare LT, %72, %71 : (tensor<15xf32>, tensor<15xf32>) -> tensor<15xi1>
      %74 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<15xf32>
      %75 = stablehlo.compare GT, %60, %74 : (tensor<15xf32>, tensor<15xf32>) -> tensor<15xi1>
      %76 = stablehlo.and %73, %75 : tensor<15xi1>
      %77 = stablehlo.select %42, %43, %61 : (tensor<15xi1>, tensor<15xf32>, tensor<15xf32>) -> tensor<15xf32>
      %78 = stablehlo.or %42, %76 : tensor<15xi1>
      %79 = stablehlo.constant dense<1> : tensor<i32>
      %80 = stablehlo.add %41, %79 : tensor<i32>
      stablehlo.return %80, %78, %77 : tensor<i32>, tensor<15xi1>, tensor<15xf32>
    }
    %81, %82 = stablehlo.rng_bit_generator %31, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<15xui32>)
    %83 = stablehlo.constant dense<9> : tensor<15xui32>
    %84 = stablehlo.shift_right_logical %82, %83 : tensor<15xui32>
    %85 = stablehlo.convert %84 : (tensor<15xui32>) -> tensor<15xf32>
    %86 = stablehlo.constant dense<1.1920929E-7> : tensor<15xf32>
    %87 = stablehlo.multiply %85, %86 : tensor<15xf32>
    %88 = stablehlo.log %44#2 : tensor<15xf32>
    %89 = stablehlo.log %87 : tensor<15xf32>
    %90 = stablehlo.divide %89, %2 : tensor<15xf32>
    %91 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<15xf32>
    %92 = stablehlo.select %6, %90, %91 : (tensor<15xi1>, tensor<15xf32>, tensor<15xf32>) -> tensor<15xf32>
    %93 = stablehlo.add %88, %92 : tensor<15xf32>
    %94 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<15xf32>
    %96 = stablehlo.subtract %93, %95 : tensor<15xf32>
    %97 = stablehlo.reshape %96 : (tensor<15xf32>) -> tensor<5x3xf32>
    %98 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %99 = stablehlo.reduce(%97 init: %98) applies stablehlo.maximum across dimensions = [1] : (tensor<5x3xf32>, tensor<f32>) -> tensor<5xf32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>
    %101 = stablehlo.subtract %97, %100 : tensor<5x3xf32>
    %102 = stablehlo.exponential %101 : tensor<5x3xf32>
    %103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %104 = stablehlo.reduce(%102 init: %103) applies stablehlo.add across dimensions = [1] : (tensor<5x3xf32>, tensor<f32>) -> tensor<5xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>
    %106 = stablehlo.divide %102, %105 : tensor<5x3xf32>
    %107 = stablehlo.reshape %106 : (tensor<5x3xf32>) -> tensor<5x3xf32>
    return %107, %81 : tensor<5x3xf32>, tensor<2xui64>
  }
}
